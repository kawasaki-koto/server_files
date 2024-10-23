from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import i2i

i2i.create_pipe("stabilityai/sdxl-turbo")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hosting():
    # クライアントから送られたデータを取得
    data = request.form
    checkpoint = data.get('checkpoint')
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt')
    steps = int(data.get('steps'))
    strength = float(data.get('strength'))
    guidance_scale = float(data.get('guidance_scale'))

    init_image = Image.open(request.files['init_image']).convert("RGB")

    output_image = i2i.create_image(checkpoint, prompt, negative_prompt, init_image, steps, strength, guidance_scale)

    img_io = io.BytesIO()
    output_image.save(img_io, 'JPEG')
    img_io.seek(0)

    # 画像をレスポンスとして返す
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    # サーバーを起動
    app.run(host='0.0.0.0', port=5000)
