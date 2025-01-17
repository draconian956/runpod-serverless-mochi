#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from pathlib import Path

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)

            with Path('message_log.log').open('a', encoding='utf8') as f:
                f.write(json.dumps(message,indent=4))
                f.write('\r\n\r\n')

            if message['type'] == 'executing':
                data = message['data']

                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == 'save_image_websocket_node':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output                

    return output_images

prompt_text = Path('Mochi_basic_api.json').read_text(encoding='utf8')

prompt = json.loads(prompt_text)
#set the text prompt for our positive CLIPTextEncode
# prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

#set the seed for our KSampler node
# prompt["3"]["inputs"]["seed"] = 5

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

prompt_id='729cb6e1-7eb2-4410-a844-1c827a232dda'
history=get_history(prompt_id)
file_indicator=history.get(prompt_id,{}).get('outputs',{}).get('40',{}).get('gifs',[])[0]
A=history.get(prompt_id,{}).get('outputs',{}).get('40',{}).get('gifs',[])[0]

# images = get_images(ws, prompt)
image=get_image(
    filename=file_indicator['filename'],
    subfolder=file_indicator['subfolder'],
    folder_type=file_indicator['type']
)
Path(file_indicator.get('filename')).write_bytes(image)
ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts

print('AAA')
#Commented out code to display the output images:

# for node_id in images:
#     for image_data in images[node_id]:
#         from PIL import Image
#         import io
#         image = Image.open(io.BytesIO(image_data))
#         image.show()

