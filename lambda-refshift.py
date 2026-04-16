import boto3
import time

def get_assumed_role_session():
    sts = boto3.client("sts")
    assumed = sts.assume_role(
        RoleArn="arn:aws:iam::ACCOUNT_B_ID:role/RedshiftCrossAccountRole",
        RoleSessionName="LambdaRedshiftSession",
        DurationSeconds=3600
    )
    creds = assumed["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name="eu-west-1"  # Account B region
    )

def lambda_handler(event, context):
    session = get_assumed_role_session()
    redshift_data = session.client("redshift-data")

    # Execute query via Data API (async)
    response = redshift_data.execute_statement(
        ClusterIdentifier="your-cluster-name",
        Database="your_db",
        DbUser="your_db_user",        # or use SecretArn
        Sql="SELECT * FROM schema.table LIMIT 10"
    )

    statement_id = response["Id"]

    # Poll for completion
    while True:
        status = redshift_data.describe_statement(Id=statement_id)
        if status["Status"] in ("FINISHED", "FAILED", "ABORTED"):
            break
        time.sleep(1)

    if status["Status"] == "FINISHED":
        result = redshift_data.get_statement_result(Id=statement_id)
        return result["Records"]
    else:
        raise Exception(f"Query failed: {status.get('Error')}")

//ption 2: Direct psycopg2 connection (requires VPC peering + port

import boto3
import psycopg2

def lambda_handler(event, context):
    # Step 1: Assume role in Account B
    sts = boto3.client("sts")
    assumed = sts.assume_role(
        RoleArn="arn:aws:iam::ACCOUNT_B_ID:role/RedshiftCrossAccountRole",
        RoleSessionName="LambdaRedshiftSession"
    )
    creds = assumed["Credentials"]

    # Step 2: Get temp DB credentials
    redshift = boto3.client(
        "redshift",
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name="eu-west-1"
    )

    db_creds = redshift.get_cluster_credentials(
        ClusterIdentifier="your-cluster-name",
        DbUser="your_db_user",
        DbName="your_db",
        AutoCreate=False,
        DurationSeconds=3600
    )

    # Step 3: Connect via psycopg2
    conn = psycopg2.connect(
        host="your-cluster.xxxx.REGION.redshift.amazonaws.com",
        port=5439,
        database="your_db",
        user=db_creds["DbUser"],
        password=db_creds["DbPassword"]
    )

    cur = conn.cursor()
    cur.execute("SELECT * FROM schema.table LIMIT 10")
    rows = cur.fetchall()
    conn.close()
    return rows

