from flask import Flask
app=Flask("mpg_prediction")

@app.route('/', methods=['GET'])

def ping():
    return "PINGING MODEL APPLICATION"


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=9696)