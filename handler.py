import json
import os
import subprocess
import boto3
import urllib
from botocore.exceptions import ClientError

def download_video_from_s3(s3_client, bucket, key, download_path):
    try:
        s3_client.download_file(bucket, key, download_path)
        print(f"Downloaded {key} to {download_path}")
    except ClientError as e:
        print(f"Error downloading {key} from S3: {e}")
        raise

def extract_frames(video_path, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            'ffmpeg',
            '-ss', '0',
            '-r', '1',
            '-i', video_path,
            '-vf', 'fps=1/10',
            '-start_number', '0',
            '-vframes', '10',
            f'{output_dir}/output-%02d.jpg',
            '-y'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Returncode:{process.returncode}\n")
            print(f"FFmpeg error:\n{stderr.decode()}")
            raise RuntimeError("FFmpeg failed to extract frames")
        print("Frames extracted successfully")

        extracted_files = os.listdir(output_dir)
        if extracted_files:
            print("Frames generated successfully:", extracted_files)
        else:
            print("No frames were generated in the output directory.")

    except Exception as e:
        print(f"Error during frame extraction: {e}")
        raise

def upload_frames_to_s3(s3_client, output_dir, output_bucket, prefix):
    print("output_dir:", output_dir)
    print("prefix:", prefix)

    prefix = prefix.rstrip('/')

    print("s3 file upload called")
    for filename in os.listdir(output_dir):
        if filename.endswith('.jpg'):
            print(filename)
            file_path = os.path.join(output_dir, filename)
            print("file_path", file_path)
            s3_key = f"{prefix}/{filename}"
            try:
                s3_client.upload_file(file_path, output_bucket, s3_key)
                print(f"Uploaded {filename} to s3://{output_bucket}/{s3_key}")
            except ClientError as e:
                print(f"Error uploading {filename} to S3: {e}")
                raise

def lambda_handler(event, context):
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    s3_client = session.client('s3')

    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    video_filename = os.path.basename(key)
    input_path = f'/tmp/{video_filename}'
    output_dir = f"/tmp/{os.path.splitext(video_filename)[0]}"

    try:
        download_video_from_s3(s3_client, bucket_name, key, input_path)

        extract_frames(input_path, output_dir)

        upload_frames_to_s3(s3_client, output_dir, OUTPUT_BUCKET, os.path.splitext(video_filename)[0])

        return {
            'statusCode': 200,
            'body': json.dumps('Frames extracted and uploaded successfully')
        }

    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing video: {str(e)}')
        }

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Removed {input_path}")
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, filename))
            os.rmdir(output_dir)
            print(f"Removed directory {output_dir}")
