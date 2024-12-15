import json
import os
import time
import urllib
import uuid
from base64 import b64encode

import requests
import runpod
from huggingface_hub import HfApi
from requests.adapters import HTTPAdapter, Retry
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils.rp_validator import validate

from schemas.api import API_SCHEMA
from schemas.download import DOWNLOAD_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.input import INPUT_SCHEMA
from schemas.interrogate import INTERROGATE_SCHEMA
from schemas.sync import SYNC_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA

LOCAL_URL = "http://127.0.0.1:8188/prompt"

BASE_URI = 'http://127.0.0.1:8188'
TIMEOUT = 600
POST_RETRIES = 3

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = int(
	os.environ.get("COMFY_POLLING_INTERVAL_MS", 250))
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = int(
	os.environ.get("COMFY_POLLING_MAX_RETRIES", 5000))
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
	'''
	Check if the service is ready to receive requests.
	'''
	while True:
		try:
			requests.get(url, timeout=120)
			return
		except requests.exceptions.RequestException:
			print("Service not ready yet. Retrying...")
		except Exception as err:
			print("Error: ", err)

		time.sleep(0.2)


def send_get_request(endpoint):
	return automatic_session.get(
		url=f'{BASE_URI}/{endpoint}',
		timeout=TIMEOUT
	)


def send_post_request(endpoint, payload, job_id, retry=0):
	response = automatic_session.post(
		url=f'{BASE_URI}/{endpoint}',
		json=payload,
		timeout=TIMEOUT
	)

	# Retry the post request in case the model has not completed loading yet
	if response.status_code == 404:
		if retry < POST_RETRIES:
			retry += 1
			logger.warn(
				f'Received HTTP 404 from endpoint: {endpoint}, Retrying: {retry}', job_id)
			time.sleep(0.2)
			send_post_request(endpoint, payload, job_id, retry)

	return response


def run_inference(inference_request):
	'''
	Run inference on a request.
	'''
	response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
								   json=inference_request, timeout=600)
	return response.json()


def validate_input(job):
	return validate(job['input'], INPUT_SCHEMA)


def validate_api(job):
	api = job['input']['api']
	api['endpoint'] = api['endpoint'].lstrip('/')

	return validate(api, API_SCHEMA)


def validate_payload(job):
	method = job['input']['api']['method']
	endpoint = job['input']['api']['endpoint']
	payload = job['input']['payload']
	validated_input = payload

	if endpoint == 'v1/sync':
		logger.info(f'Validating /{endpoint} payload', job['id'])
		validated_input = validate(payload, SYNC_SCHEMA)
	elif endpoint == 'v1/download':
		logger.info(f'Validating /{endpoint} payload', job['id'])
		validated_input = validate(payload, DOWNLOAD_SCHEMA)
	elif endpoint == 'sdapi/v1/txt2img':
		logger.info(f'Validating /{endpoint} payload', job['id'])
		validated_input = validate(payload, TXT2IMG_SCHEMA)
	elif endpoint == 'sdapi/v1/img2img':
		logger.info(f'Validating /{endpoint} payload', job['id'])
		validated_input = validate(payload, IMG2IMG_SCHEMA)
	elif endpoint == 'sdapi/v1/interrogate' and method == 'POST':
		logger.info(f'Validating /{endpoint} payload', job['id'])
		validated_input = validate(payload, INTERROGATE_SCHEMA)

	return endpoint, job['input']['api']['method'], validated_input


def download(job):
	source_url = job['input']['payload']['source_url']
	download_path = job['input']['payload']['download_path']
	process_id = os.getpid()
	temp_path = f"{download_path}.{process_id}"

	# Download the file and save it as a temporary file
	with requests.get(source_url, stream=True) as r:
		r.raise_for_status()
		with open(temp_path, 'wb') as f:
			for chunk in r.iter_content(chunk_size=8192):
				f.write(chunk)

	# Rename the temporary file to the actual file name
	os.rename(temp_path, download_path)
	logger.info(
		f'{source_url} successfully downloaded to {download_path}', job['id'])

	return {
		'msg': 'Download successful',
		'source_url': source_url,
		'download_path': download_path
	}


def sync(job):
	repo_id = job['input']['payload']['repo_id']
	sync_path = job['input']['payload']['sync_path']
	hf_token = job['input']['payload']['hf_token']

	api = HfApi()

	models = api.list_repo_files(
		repo_id=repo_id,
		token=hf_token
	)

	synced_count = 0
	synced_files = []

	for model in models:
		folder = os.path.dirname(model)
		dest_path = f'{sync_path}/{model}'

		if folder and not os.path.exists(dest_path):
			logger.info(f'Syncing {model} to {dest_path}', job['id'])

			uri = api.hf_hub_download(
				token=hf_token,
				repo_id=repo_id,
				filename=model,
				local_dir=sync_path,
				local_dir_use_symlinks=False
			)

			if uri:
				synced_count += 1
				synced_files.append(dest_path)

	return {
		'synced_count': synced_count,
		'synced_files': synced_files
	}


def queue_prompt(prompt):
	p = {"prompt": prompt, "client_id": client_id}
	data = json.dumps(p).encode('utf-8')
	req = urllib.request.Request(
		"http://{}/prompt".format(server_address), data=data)
	return json.loads(urllib.request.urlopen(req).read())


def get_file_content(filename, subfolder, folder_type):
	data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
	url_values = urllib.parse.urlencode(data)
	with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
		return response.read()


def get_history(prompt_id):
	with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
		return json.loads(response.read())

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
# def handler(event):
# 	'''
# 	This is the handler function that will be called by the serverless.
# 	'''

# 	json = run_inference(event["input"])

# 	# return the output that you want to be returned like pre-signed URLs to output artifacts
# 	return json


def handler(job):
	try:
		queued_workflow = queue_prompt(job['input']['payload'])
		prompt_id = queued_workflow['prompt_id']
		print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")
	except Exception as e:
		return {"error": f"Error queuing workflow: {str(e)}"}

	# Poll for completion
	print("runpod-worker-comfy - wait until workflow is complete")
	retries = 0
	try:
		while retries < COMFY_POLLING_MAX_RETRIES:
			history = get_history(prompt_id)

			# Exit the loop if we have found the history
			if prompt_id in history and history[prompt_id].get("outputs"):
				break
			else:
				# Wait before trying again
				time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
				retries += 1
		else:
			return {"error": "Max retries reached while waiting for video generation"}
	except Exception as e:
		return {"error": f"Error waiting for video generation: {str(e)}"}

	file_indicator = (
		history.get(prompt_id, {})
		.get('outputs', {})
		.get('40', {})
		.get('gifs', [])
	)[0]

	video_bytes = get_file_content(
		filename=file_indicator['filename'],
		subfolder=file_indicator['subfolder'],
		folder_type=file_indicator['type']
	)
	video_base64_bytes = b64encode(video_bytes)
	video_base64_str = video_base64_bytes.decode('utf8')
	print(
		"runpod-worker-comfy - the file was generated and converted to base64"
	)

	return {
		'file_indicator': file_indicator,
		'video_base64': video_base64_str,
	}


if __name__ == "__main__":
	wait_for_service(f'{BASE_URI}/models')
	logger.info('ComfyUI API is ready')
	logger.info('Starting RunPod Serverless...')
	runpod.serverless.start(
		{
			'handler': handler
		}
	)
