import datetime
import boto3
from airflow.decorators import dag
@dag(start_date=datetime.datetime(2021, 1, 1), schedule="@daily", catchup=False, tags=["training", "ssm", "ec2"])
def trigger_model_training():
    try:
        ssm = boto3.client('ssm', region_name='ca-central-1')
        response = ssm.send_command(
            InstanceIds=['i-0378e09df9174f371'],
            DocumentName="AWS-RunShellScript",
            Parameters={
                'commands': [
                    'cd /home/ec2-user/dana-4830/scripts',
                    'python3 train_model_nca_xgb_deg.py > output1.log 2>&1'
                ]
            },
        )
        command_id = response['Command']['CommandId']
        print(f"SSM command sent: Command ID = {command_id}")
        return command_id
    except Exception as e:
        print(f"Error triggering EC2 via SSM: {str(e)}")
        raise
trigger_model_training()