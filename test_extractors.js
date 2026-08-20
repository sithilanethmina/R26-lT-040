const fs = require('fs');
global.window = {};

eval(fs.readFileSync('browser-extension/extractors/gpu.js', 'utf8'));
eval(fs.readFileSync('browser-extension/extractors/mobile.js', 'utf8'));
eval(fs.readFileSync('browser-extension/extractors/vehicle.js', 'utf8'));
eval(fs.readFileSync('browser-extension/extractors/electronics.js', 'utf8'));

const testCases = [
  { title: 'ASUS KO Geforce RTX 3060 TI V2 OC Edition 8GB GDDR6 Brand New VGA', price: 123000 },
  { title: 'GTX 1650 4GB VGA Card', price: 41500 },
  { title: 'Asus Dual RTX 4070 Ti Super 16GB OC VGA Card', price: 295000 },
  { title: 'RX 580 8GB Asrock VGA', price: 31000 },
  { title: 'GTX 1080 Super 8GB Gaming VGA', price: 65000 },
  { title: 'Gaming VGA Card for Sale', price: 50000 } // Missing model -> Should prompt manual
];

console.log('--- TESTING GPU EXTRACTOR ---');
testCases.forEach((tc, idx) => {
  const parsed = global.window.FairPriceLK_Extractors.gpu.parse({ title: tc.title, price: tc.price });
  console.log(`[Test ${idx+1}] Valid: ${parsed.valid} | Model: ${parsed.data.model} | VRAM: ${parsed.data.vram_gb}GB | Brand: ${parsed.data.brand} | Error: ${parsed.error_message || 'None'}`);
});
