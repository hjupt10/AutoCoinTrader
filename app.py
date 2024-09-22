# app.py
from flask import Flask, render_template, jsonify, request
from AUTO import TradingBot
import threading
import traceback
import logging

app = Flask(__name__)
bot = TradingBot()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Start the TradingBot in a separate thread to run concurrently with Flask
def run_bot():
    bot.run()

bot_thread = threading.Thread(target=run_bot)
bot_thread.daemon = True
bot_thread.start()

@app.route('/')
def index():
    return render_template('index.html', coins=bot.get_available_coins())

@app.route('/api/select_coins', methods=['POST'])
def select_coins():
    selected_coins = request.json.get('coins', [])
    bot.select_coins(selected_coins)
    return jsonify({"status": "success"})

@app.route('/api/get_discord_messages')
def get_discord_messages():
    try:
        messages = bot.get_discord_messages()
        return jsonify(messages)
    except Exception as e:
        app.logger.error(f"Error in get_discord_messages: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_trade_history')
def get_trade_history():
    try:
        history = bot.get_trade_history()
        return jsonify(history)
    except Exception as e:
        app.logger.error(f"Error in get_trade_history: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_performance_data')
def get_performance_data():
    try:
        data = bot.get_performance_data()
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error in get_performance_data: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_trading_status')
def get_trading_status():
    try:
        status = bot.get_trading_status()
        return jsonify({"status": status})
    except Exception as e:
        app.logger.error(f"Error in get_trading_status: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
