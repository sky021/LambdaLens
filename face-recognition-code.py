import json
import boto3
import urllib
import os
from botocore.exceptions import ClientError
from PIL import Image
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import tempfile

# Override the default model directory to use /tmp
os.environ['TORCH_HOME'] = '/tmp'

# Create necessary directories in /tmp
model_dir = os.path.join('/tmp', '.cache', 'torch', 'checkpoints')
os.makedirs(model_dir, exist_ok=True)

def initialize_models():
    try:
        # Initialize MTCNN
        mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
        
        # Initialize InceptionResnetV1 with vggface2
        with tempfile.TemporaryDirectory(dir='/tmp') as temp_dir:
            # Temporarily set the model directory
            old_torch_home = os.environ.get('TORCH_HOME')
            os.environ['TORCH_HOME'] = temp_dir
            
            # Initialize the model
            resnet = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Restore original TORCH_HOME
            if old_torch_home:
                os.environ['TORCH_HOME'] = old_torch_home
            else:
                del os.environ['TORCH_HOME']
        
        return mtcnn, resnet
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

def download_image_from_s3(s3_client, bucket, key, download_path):
    try:
        s3_client.download_file(bucket, key, download_path)
        print(f"Downloaded {key} to {download_path}")
    except ClientError as e:
        print(f"Error downloading {key} from S3: {e}")
        raise

def process_face_recognition(image_path, data_path, mtcnn, resnet):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        boxes, _ = mtcnn.detect(img)
        
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        face, prob = mtcnn(img, return_prob=True, save_path=None)
        
        print("Loading the data.pt file")
        saved_data = torch.load(data_path)
        
        result_text = "No face detected"
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach()
            embedding_list = saved_data[0]
            name_list = saved_data[1]
            
            dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
            idx_min = dist_list.index(min(dist_list))
            
            result_text = name_list[idx_min]
        
        return result_text
    
    except Exception as e:
        print(f"Error during face recognition: {e}")
        raise

def upload_result_to_s3(s3_client, path, output_bucket, key):
    try:
        s3_client.upload_file(path, output_bucket, f'{key}.txt')
        print(f"Uploaded result to s3://{output_bucket}/{key}.txt")
    except ClientError as e:
        print(f"Error uploading result to S3: {e}")
        raise

def face_recognition(event, context):
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    s3_client = session.client('s3')
    
    try:
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        print(f"Processing image from bucket: {bucket_name}, key: {key}")
        
        image_filename = key
        name = image_filename[:-4]
        input_path = f'/tmp/{image_filename}'
        result_path = f'/tmp/{name}.txt'
        data_pt_path = '/tmp/data.pt'
        data_bucket_name = 'cse546-pj3-data'
        
        # Initialize models first
        print("Initializing models...")
        mtcnn, resnet = initialize_models()
        print("Models initialized successfully")
        
        # Download both the image and data.pt file
        print("Downloading files from S3...")
        download_image_from_s3(s3_client, bucket_name, key, input_path)
        download_image_from_s3(s3_client, data_bucket_name, 'data.pt', data_pt_path)
        print("Files downloaded successfully")
        
        # Process the image
        print("Processing face recognition...")
        result_text = process_face_recognition(input_path, data_pt_path, mtcnn, resnet)
        print(f"Recognition result: {result_text}")
        
        # Write and upload result
        with open(result_path, 'w+') as f:
            f.write(result_text)
        
        output_bucket = '1231868809-output'
        upload_result_to_s3(s3_client, result_path, output_bucket, name)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'recognized_name': result_text,
                'output_file': f'{name}.txt',
                'output_bucket': output_bucket
            })
        }
    
    except Exception as e:
        print(f"Error in face_recognition handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing image: {str(e)}')
        }
    
    finally:
        # Clean up temporary files
        temp_files = [
            input_path, 
            result_path, 
            data_pt_path
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    print(f"Error removing temporary file {temp_file}: {e}")