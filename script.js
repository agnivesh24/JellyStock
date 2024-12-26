// Chart initialization for dashboard
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('stockChart')) {
        fetch('/api/inventory')
            .then(response => response.json())
            .then(data => {
                const ctx = document.getElementById('stockChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: data.map(item => item.name),
                        datasets: [{
                            label: 'Current Stock',
                            data: data.map(item => item.quantity),
                            backgroundColor: '#2563eb',
                            borderColor: '#1e40af',
                            borderWidth: 1
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
            });
    }
});

// Form validation
const itemForm = document.querySelector('.item-form');
if (itemForm) {
    itemForm.addEventListener('submit', function(e) {
        const quantity = document.getElementById('quantity').value;
        const price = document.getElementById('price').value;
        
        if (quantity < 0 || price < 0) {
            e.preventDefault();
            alert('Quantity and price must be positive numbers');
        }
    });
} 

// Intersection Observer for animations
document.addEventListener('DOMContentLoaded', function() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-fadeInUp').forEach((element) => {
        observer.observe(element);
    });

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}); 