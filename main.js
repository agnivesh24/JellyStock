// Demo Modal Functionality
function openDemo() {
    document.getElementById('demoModal').style.display = 'block';
}

document.addEventListener('DOMContentLoaded', function() {
    // Close modal when clicking the X
    document.querySelector('.close-modal').onclick = function() {
        document.getElementById('demoModal').style.display = 'none';
    }

    // Close modal when clicking outside
    window.onclick = function(event) {
        if (event.target == document.getElementById('demoModal')) {
            document.getElementById('demoModal').style.display = 'none';
        }
    }

    // Add new inventory item
    document.getElementById('addItem').onclick = function() {
        const newItem = document.createElement('div');
        newItem.className = 'inventory-item';
        newItem.innerHTML = `
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
            <select class="item-type" required>
                <option value="">Select Type</option>
                <option value="Electronics">Electronics</option>
                <option value="Accessories">Accessories</option>
                <option value="Peripherals">Peripherals</option>
                <option value="Components">Components</option>
            </select>
            <input type="number" placeholder="Current stock" required>
            <input type="number" placeholder="Price" required>
            <button type="button" class="remove-item">&times;</button>
        `;
        document.getElementById('inventoryList').appendChild(newItem);
    }

    // Remove inventory item
    document.getElementById('inventoryList').onclick = function(e) {
        if (e.target.classList.contains('remove-item')) {
            e.target.parentElement.remove();
        }
    }

    // Handle form submission
    document.getElementById('demoForm').onsubmit = function(e) {
        e.preventDefault();
        alert('Stock prediction complete!');
    }
}); 