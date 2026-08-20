const fs = require('fs');

let fileContent = fs.readFileSync('browser-extension/extractors/gpu.js', 'utf8');
fileContent = fileContent.replace('function normalizeText', 'global.normalizeText = function normalizeText');
fileContent = fileContent.replace('const CANONICAL_MODELS =', 'global.CANONICAL_MODELS =');
global.window = {};
eval(fileContent);

const title = 'ASUS KO Geforce RTX 3060 TI V2 OC Edition 8GB GDDR6 Brand New VGA';
const normTitle = global.normalizeText(title);
const padded = ` ${normTitle} `;

console.log('Padded:', padded);

const sorted = [...global.CANONICAL_MODELS].sort((a, b) => global.normalizeText(b).length - global.normalizeText(a).length);

console.log('First 20 sorted canonical models:');
sorted.slice(0, 20).forEach((m, idx) => {
    const normM = global.normalizeText(m);
    const regex = new RegExp(`(?:^|\\s)${normM.replace(/\\s+/g, '\\s+')}(?=\\s|$)`, 'i');
    const matches = regex.test(padded);
    console.log(`[${idx+1}] ${m} (norm: "${normM}", len: ${normM.length}) -> matches: ${matches}`);
});


