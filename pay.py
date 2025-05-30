from flask import Flask, request, jsonify, send_file, send_from_directory
import razorpay
import hmac
import hashlib

app = Flask(__name__)

# Replace with your live Razorpay credentials
RAZORPAY_KEY_ID = "rzp_live_Z4vJOV2SS1Fq2F"
RAZORPAY_KEY_SECRET = "jzBBpsxmPKWB95rVVMXCu6LY"

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/policy')
def policy():
    return send_file('policy.html')

@app.route('/create-order', methods=['POST'])
def create_order():
    data = request.get_json()
    amount_rupees = data.get('amount', 1)  # default ₹1
    amount_paise = int(amount_rupees * 100)

    razorpay_order = razorpay_client.order.create(dict(
        amount=amount_paise,
        currency='INR',
        payment_capture=1
    ))

    return jsonify({
        'order_id': razorpay_order['id'],
        'amount': razorpay_order['amount'],
        'currency': razorpay_order['currency'],
        'key_id': RAZORPAY_KEY_ID
    })

@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    data = request.get_json()

    generated_signature = hmac.new(
        bytes(RAZORPAY_KEY_SECRET, 'utf-8'),
        bytes(data['razorpay_order_id'] + "|" + data['razorpay_payment_id'], 'utf-8'),
        hashlib.sha256
    ).hexdigest()

    if generated_signature == data['razorpay_signature']:
        return jsonify({'message': '✅ Payment verified successfully'}), 200
    else:
        return jsonify({'message': '❌ Payment verification failed'}), 400

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
