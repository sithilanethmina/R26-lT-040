/* Modern SaaS Dashboard Prediction Script */

function updateBrandDropdown(category) {
    const brandSelect = document.getElementById('brand');
    brandSelect.innerHTML = '<option value="" disabled selected>Select Brand</option>';

    let brands = [];
    if (category === 'laptop') {
        brands = ['DELL', 'HP', 'ASUS', 'ACER', 'APPLE'];
    } else if (category === 'monitor') {
        brands = ['SAMSUNG', 'DELL', 'HP', 'LG', 'ASUS', 'ACER', 'VIEWSONIC', 'BENQ'];
    } else if (category === 'tablet') {
        brands = ['APPLE', 'SAMSUNG', 'HUAWEI', 'AMAZON', 'XIAOMI', 'MICROSOFT'];
    }

    brands.forEach(brand => {
        const option = document.createElement('option');
        option.value = brand;
        option.textContent = brand;
        brandSelect.appendChild(option);
    });
}

function updateModelDropdown(category, brand) {
    const modelSelect = document.getElementById('model');
    modelSelect.innerHTML = '<option value="Other" selected>Other / Generic</option>';

    const laptopModels = {
        'DELL': ['LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15'],
        'HP': ['ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'OMEN', 'VICTUS'],
        'LENOVO': ['THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'V15'],
        'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF'],
        'ACER': ['ASPIRE', 'SWIFT', 'NITRO', 'PREDATOR'],
        'APPLE': ['MACBOOK PRO', 'MACBOOK AIR'],
        'MSI': ['MODERN', 'STEALTH', 'KATANA', 'GAMING']
    };

    const tabletModels = {
        'APPLE': ['IPAD PRO', 'IPAD AIR', 'IPAD MINI', 'IPAD'],
        'SAMSUNG': ['GALAXY TAB S', 'GALAXY TAB A', 'GALAXY TAB E'],
        'HUAWEI': ['MATEPAD', 'MEDIAPAD'],
        'LENOVO': ['TAB P11', 'TAB M10', 'YOGA TAB'],
        'XIAOMI': ['MI PAD', 'PAD 5', 'PAD 6']
    };

    if (category === 'laptop' && laptopModels[brand]) {
        laptopModels[brand].forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    } else if (category === 'tablet' && tabletModels[brand]) {
        tabletModels[brand].forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    }
}

document.getElementById('brand').addEventListener('change', (e) => {
    const category = document.getElementById('category').value;
    updateModelDropdown(category, e.target.value);
});

// Handle Radio Button Changes
document.querySelectorAll('input[name="category-radio"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        const category = e.target.value;
        document.getElementById('category').value = category;
        
        const laptopFields = document.getElementById('laptop-fields');
        const monitorFields = document.getElementById('monitor-fields');
        const tabletFields = document.getElementById('tablet-fields');
        const modelGroup = document.getElementById('model-group');
        const gbOption = document.getElementById('gb-option');
        
        updateBrandDropdown(category);
        
        laptopFields.style.display = 'none';
        monitorFields.style.display = 'none';
        tabletFields.style.display = 'none';
        
        if (category === 'laptop') {
            laptopFields.style.display = 'block';
            modelGroup.style.display = 'block';
            gbOption.style.display = 'block';
        } else if (category === 'monitor') {
            monitorFields.style.display = 'block';
            modelGroup.style.display = 'none';
            gbOption.style.display = 'none';
        } else {
            tabletFields.style.display = 'block';
            modelGroup.style.display = 'block';
            gbOption.style.display = 'none';
        }
    });
});

// Prediction Form Submission
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const resultBox = document.getElementById('result-box');
    const welcomeBox = document.getElementById('welcome-box');
    const errorBox = document.getElementById('error-box');
    const btnText = document.querySelector('#predict-btn span');
    const loader = document.getElementById('loader');

    // UI Feedback
    btnText.style.display = 'none';
    loader.style.display = 'block';
    errorBox.classList.add('hidden');
    resultBox.classList.remove('show-result');

    const category = document.getElementById('category').value;
    const algorithm = document.getElementById('algorithm').value;
    const brand = document.getElementById('brand').value;
    const model = document.getElementById('model').value;

    const data = {
        category: category,
        algorithm: algorithm,
        brand: brand,
        model: model,
        condition: 'Used'
    };

    if (category === 'laptop') {
        data.cpu = document.getElementById('cpu').value;
        data.generation = document.getElementById('generation').value;
        data.ram = document.getElementById('ram').value;
        data.storage = document.getElementById('storage').value;
        data.storageType = document.getElementById('storageType').value;
    } else if (category === 'monitor') {
        data.size = document.getElementById('size').value;
        data.refreshRate = document.getElementById('hz').value;
        data.resolution = document.getElementById('resolution').value;
        data.condition = 'Used'; // Monitors currently default to Used
    } else if (category === 'tablet') {
        data.ram = document.getElementById('tab-ram').value;
        data.storage = document.getElementById('tab-storage').value;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok && result.success) {
            document.getElementById('price-result').innerText = result.price;
            document.getElementById('active-model-name').innerText = result.model_name;
            
            const modelList = document.getElementById('model-list');
            modelList.innerHTML = '';
            
            Object.keys(result.all_results).forEach(modelName => {
                const data = result.all_results[modelName];
                const accuracy = (data.R2 * 100).toFixed(1);
                const isActive = modelName === result.model_name;
                
                const modelItem = document.createElement('div');
                modelItem.style.marginBottom = '1.2rem';
                modelItem.innerHTML = `
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.4rem;">
                        <span style="${isActive ? 'color: white; font-weight: 600;' : 'color: var(--text-muted);'}">
                            ${modelName} ${isActive ? '<span style="font-size: 0.7rem; background: var(--primary); padding: 1px 4px; border-radius: 4px; margin-left: 5px;">ACTIVE</span>' : ''}
                        </span>
                        <div style="text-align: right;">
                            <span style="color: ${isActive ? 'var(--success)' : 'var(--text-muted)'}; font-weight: 600; display: block;">${accuracy}%</span>
                            <span style="font-size: 0.7rem; color: var(--text-muted); opacity: 0.8;">R² Score: ${data.R2.toFixed(4)}</span>
                        </div>
                    </div>
                    <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; width: ${accuracy}%; background: ${isActive ? 'var(--success)' : 'var(--text-muted)'}; border-radius: 2px; transition: width 1s ease;"></div>
                    </div>
                `;
                modelList.appendChild(modelItem);
            });
            
            welcomeBox.classList.add('hidden');
            resultBox.classList.remove('hidden');
            setTimeout(() => {
                resultBox.classList.add('show-result');
            }, 50);
        } else {
            errorBox.innerText = result.error || 'An error occurred while predicting.';
            errorBox.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Error:', error);
        errorBox.innerText = 'Failed to connect to the prediction server.';
        errorBox.classList.remove('hidden');
    } finally {
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
});

// Initialize
updateBrandDropdown('laptop');
