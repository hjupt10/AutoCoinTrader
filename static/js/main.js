// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    const coinButtons = document.querySelectorAll('.coin-btn');
    const performanceData = document.getElementById('performance-data');
    const tradeTable = document.getElementById('trade-table').getElementsByTagName('tbody')[0];
    const discordMessageList = document.getElementById('discord-message-list');
    let profitChart;

    // Add event listeners to coin buttons
    coinButtons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.toggle('selected');
            updateSelectedCoins();
        });
    });

    function updateSelectedCoins() {
        const selectedCoins = Array.from(document.querySelectorAll('.coin-btn.selected')).map(btn => btn.dataset.coin);
        fetch('/api/select_coins', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({coins: selectedCoins}),
        })
        .then(response => response.json())
        .then(data => {
            if(data.status !== 'success') {
                console.error('Failed to select coins:', data);
            }
        })
        .catch(error => {
            console.error('Error selecting coins:', error);
        });
    }

    function updatePerformance() {
        fetch('/api/get_performance_data')
            .then(response => response.json())
            .then(data => {
                performanceData.innerHTML = `
                    <p>Total Profit: ${data.total_profit.toFixed(2)} KRW</p>
                    <p>Total Profit %: ${data.total_profit_pct.toFixed(2)}%</p>
                    <p>Win Rate: ${data.win_rate.toFixed(2)}%</p>
                    <p>Average Profit: ${data.avg_profit.toFixed(2)} KRW</p>
                    <p>Sharpe Ratio: ${data.sharpe_ratio.toFixed(2)}</p>
                    <p>Max Drawdown: ${data.max_drawdown.toFixed(2)}%</p>
                `;

                updateProfitChart(data.total_profit_pct);
            })
            .catch(error => {
                console.error('Error fetching performance data:', error);
            });
    }

    function updateTradeHistory() {
        fetch('/api/get_trade_history')
            .then(response => response.json())
            .then(data => {
                tradeTable.innerHTML = '';
                data.forEach(trade => {
                    let row = tradeTable.insertRow();
                    row.innerHTML = `
                        <td>${trade.timestamp}</td>
                        <td>${trade.symbol}</td>
                        <td>${trade.order_type}</td>
                        <td>${trade.price}</td>
                        <td>${trade.quantity}</td>
                        <td>${trade.status}</td>
                    `;
                });
            })
            .catch(error => {
                console.error('Error fetching trade history:', error);
            });
    }

    function updateDiscordMessages() {
        fetch('/api/get_discord_messages')
            .then(response => response.json())
            .then(messages => {
                discordMessageList.innerHTML = '';
                messages.forEach(message => {
                    let li = document.createElement('li');
                    li.textContent = message;
                    discordMessageList.appendChild(li);
                });
            })
            .catch(error => {
                console.error('Error fetching Discord messages:', error);
            });
    }

    function updateProfitChart(profitPct) {
        const ctx = document.getElementById('profit-chart').getContext('2d');
        
        if (profitChart) {
            profitChart.destroy();
        }

        profitChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Initial', 'Current'],
                datasets: [{
                    label: 'Profit Percentage',
                    data: [0, profitPct],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    function updateTradingStatus() {
        fetch('/api/get_trading_status')
            .then(response => response.json())
            .then(data => {
                const statusIndicator = document.getElementById('status-indicator');
                if (data.status === 'active') {
                    statusIndicator.className = 'status-indicator status-active';
                    statusIndicator.title = 'Trading Active';
                } else {
                    statusIndicator.className = 'status-indicator status-inactive';
                    statusIndicator.title = 'Trading Inactive';
                }
            })
            .catch(error => console.error('Error fetching trading status:', error));
    }

    // 페이지 로드 시 초기 업데이트
    updateTradingStatus();

    // 10초마다 거래 상태 업데이트
    setInterval(updateTradingStatus, 10000);

    // Initial update
    updatePerformance();
    updateTradeHistory();
    updateDiscordMessages();

    // Update every 60 seconds
    setInterval(() => {
        updatePerformance();
        updateTradeHistory();
        updateDiscordMessages();
    }, 60000);
});
