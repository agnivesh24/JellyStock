document.addEventListener('DOMContentLoaded', function() {
    const addItemBtn = document.getElementById('addItem');
    const inventoryList = document.getElementById('inventoryList');
    const demoForm = document.getElementById('demoForm');
    const predictionResults = document.getElementById('predictionResults');

    addItemBtn.onclick = function() {
        const newItem = document.createElement('div');
        newItem.className = 'inventory-item';
        newItem.innerHTML = `
            <select class="item-type" required>
                <option value="">Select Type</option>
                <option value="Electronics">Electronics</option>
                <option value="Accessories">Accessories</option>
                <option value="Peripherals">Peripherals</option>
                <option value="Components">Components</option>
            </select>
            <select class="item-name" required>
                <option value="">Select Item</option>
                <option value="Laptop">Laptop</option>
                <option value="Smartphone">Smartphone</option>
                <option value="Tablet">Tablet</option>
                <option value="Headphones">Headphones</option>
                <option value="Monitor">Monitor</option>
                <option value="Keyboard">Keyboard</option>
                <option value="Mouse">Mouse</option>
            </select>
            <input type="number" placeholder="Current stock" required>
            <input type="number" placeholder="Price" required>
            <div class="monthly-sales">
                <h4>Last 12 Months Sales</h4>
                <div class="sales-inputs">
                    <input type="number" placeholder="Jan" required>
                    <input type="number" placeholder="Feb" required>
                    <input type="number" placeholder="Mar" required>
                    <input type="number" placeholder="Apr" required>
                    <input type="number" placeholder="May" required>
                    <input type="number" placeholder="Jun" required>
                    <input type="number" placeholder="Jul" required>
                    <input type="number" placeholder="Aug" required>
                    <input type="number" placeholder="Sep" required>
                    <input type="number" placeholder="Oct" required>
                    <input type="number" placeholder="Nov" required>
                    <input type="number" placeholder="Dec" required>
                </div>
            </div>
            <button type="button" class="remove-item">&times;</button>
        `;
        inventoryList.appendChild(newItem);
    };

    demoForm.onsubmit = function(e) {
        e.preventDefault();
        const items = document.querySelectorAll('.inventory-item');
        const resultsGrid = document.querySelector('.results-grid');
        resultsGrid.innerHTML = '';

        items.forEach(item => {
            const name = item.querySelector('.item-name').value;
            const type = item.querySelector('.item-type').value;
            const currentStock = parseInt(item.querySelector('input[type="number"]').value);
            const price = parseFloat(item.querySelector('input[type="number"]:nth-of-type(2)').value);
            const monthlySales = Array.from(item.querySelectorAll('.sales-inputs input'))
                .map(input => parseInt(input.value));
            
            const prediction = predictNextMonth(monthlySales, currentStock, price);
            
            const resultCard = document.createElement('div');
            resultCard.className = `prediction-card ${prediction.status.toLowerCase()}`;
            resultCard.innerHTML = `
                <h3>${name}</h3>
                <p class="item-type">${type}</p>
                <div class="prediction-details">
                    <div class="detail">
                        <span>Current Stock:</span>
                        <strong>${currentStock}</strong>
                    </div>
                    <div class="detail">
                        <span>Predicted Next Month:</span>
                        <strong>${prediction.nextMonth}</strong>
                    </div>
                    <div class="detail">
                        <span>Recommended Order:</span>
                        <strong>${prediction.orderAmount}</strong>
                    </div>
                    <div class="detail">
                        <span>Confidence Level:</span>
                        <strong>${prediction.confidence}%</strong>
                    </div>
                    <div class="status ${prediction.status.toLowerCase()}">
                        ${prediction.status}
                    </div>
                </div>
                <div class="trend-analysis">
                    <h4>Trend Analysis</h4>
                    <p>${prediction.trendAnalysis}</p>
                </div>
                <div class="chart">
                    ${generateChart(monthlySales, prediction.nextMonth)}
                </div>
            `;
            resultsGrid.appendChild(resultCard);
        });

        predictionResults.style.display = 'block';
        predictionResults.scrollIntoView({ behavior: 'smooth' });
    };
});

function predictNextMonth(monthlySales, currentStock, price) {
    // Calculate moving average
    const movingAverage = monthlySales.slice(-3).reduce((a, b) => a + b, 0) / 3;
    
    // Calculate seasonal index
    const monthIndex = new Date().getMonth();
    const lastYearSameMonth = monthlySales[monthIndex];
    const yearAverage = monthlySales.reduce((a, b) => a + b, 0) / 12;
    const seasonalIndex = lastYearSameMonth / yearAverage;
    
    // Calculate trend
    const firstHalf = monthlySales.slice(0, 6).reduce((a, b) => a + b, 0) / 6;
    const secondHalf = monthlySales.slice(-6).reduce((a, b) => a + b, 0) / 6;
    const trend = secondHalf - firstHalf;
    
    // Predict next month
    let predictedSales = Math.round(movingAverage * seasonalIndex * (1 + trend/100));
    
    // Calculate confidence level based on data consistency
    const stdDev = calculateStandardDeviation(monthlySales);
    const coefficient = (stdDev / yearAverage) * 100;
    const confidence = Math.round(Math.max(0, 100 - coefficient));
    
    // Determine status and order amount
    let status = 'Sufficient';
    let orderAmount = 0;
    
    if (currentStock < predictedSales) {
        status = currentStock < predictedSales/2 ? 'Critical' : 'Low';
        orderAmount = Math.ceil(predictedSales * 1.2 - currentStock); // 20% buffer
    }
    
    // Generate trend analysis text
    const trendAnalysis = generateTrendAnalysis(trend, seasonalIndex, monthlySales);
    
    return {
        nextMonth: predictedSales,
        orderAmount: orderAmount,
        status: status,
        confidence: confidence,
        trendAnalysis: trendAnalysis
    };
}

function calculateStandardDeviation(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squareDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquareDiff);
}

function generateTrendAnalysis(trend, seasonalIndex, monthlySales) {
    const lastThreeMonthsTrend = 
        monthlySales[monthlySales.length - 1] > monthlySales[monthlySales.length - 2] &&
        monthlySales[monthlySales.length - 2] > monthlySales[monthlySales.length - 3];
    
    let analysis = '';
    
    if (trend > 0) {
        analysis += 'Upward trend detected. ';
    } else if (trend < 0) {
        analysis += 'Downward trend detected. ';
    } else {
        analysis += 'Stable sales pattern. ';
    }
    
    if (seasonalIndex > 1.1) {
        analysis += 'Historically strong month. ';
    } else if (seasonalIndex < 0.9) {
        analysis += 'Historically slow month. ';
    }
    
    if (lastThreeMonthsTrend) {
        analysis += 'Recent months show increasing demand.';
    }
    
    return analysis;
}

function generateChart(monthlySales, predictedSales) {
    const allValues = [...monthlySales, predictedSales];
    const max = Math.max(...allValues);
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Next'];
    
    return allValues.map((value, index) => {
        const height = (value / max * 100);
        const isPrediction = index === allValues.length - 1;
        return `
            <div class="bar-container">
                <div class="bar ${isPrediction ? 'prediction' : ''}" 
                     style="height: ${height}%" 
                     title="${months[index]}: ${value} units">
                </div>
                <span class="month-label">${months[index]}</span>
            </div>
        `;
    }).join('');
} 