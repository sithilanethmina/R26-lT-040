/**
 * FairPriceLK - GPU Category Extractor
 * Powered by comprehensive GPU specifications dataset (trusted_gpu_specs & gpu_specs reference).
 * Supports both full canonical names (e.g., 'GTX 970', 'RTX 3060', 'RX 580')
 * and prefix-omitted standalone model numbers (e.g., 'Zotac 970 4GB' -> 'GTX 970').
 * Automatically auto-fills standard VRAM when omitted, validates against official specifications,
 * and returns clear error messages for invalid specs.
 */

window.FairPriceLK_Extractors = window.FairPriceLK_Extractors || {};

window.FairPriceLK_Extractors.gpu = (function () {

    const KNOWN_BRANDS = [
        "ASUS", "MSI", "GIGABYTE", "ZOTAC", "GALAX", "PALIT", "SAPPHIRE",
        "ASROCK", "POWERCOLOR", "COLORFUL", "INNO3D", "PNY", "EVGA", "EMTEK",
        "GAINWARD", "XFX", "MANLI", "LEADTEK", "NVIDIA", "AMD", "INTEL"
    ];

    const NVIDIA_ONLY_BRANDS = ["ZOTAC", "GALAX", "EVGA", "INNO3D", "PALIT", "GAINWARD", "PNY", "MANLI", "LEADTEK", "NVIDIA"];
    const AMD_ONLY_BRANDS = ["SAPPHIRE", "POWERCOLOR", "XFX", "ASROCK", "AMD"];

    // Standalone model number rules (sorted by specificity) to map titles like 'Zotac 970 4GB' -> 'GTX 970'
    const STANDALONE_MODEL_RULES = [
        // NVIDIA RTX 40 series
        { pattern: /\b4090\b/i, nvidia: "RTX 4090" },
        { pattern: /\b4080\s*SUPER\b/i, nvidia: "RTX 4080 SUPER" },
        { pattern: /\b4080\b/i, nvidia: "RTX 4080" },
        { pattern: /\b4070\s*TI\s*SUPER\b/i, nvidia: "RTX 4070 TI SUPER" },
        { pattern: /\b4070\s*TI\b/i, nvidia: "RTX 4070 TI" },
        { pattern: /\b4070\s*SUPER\b/i, nvidia: "RTX 4070 SUPER" },
        { pattern: /\b4070\b/i, nvidia: "RTX 4070" },
        { pattern: /\b4060\s*TI\b/i, nvidia: "RTX 4060 TI" },
        { pattern: /\b4060\b/i, nvidia: "RTX 4060" },

        // NVIDIA RTX 30 series
        { pattern: /\b3090\s*TI\b/i, nvidia: "RTX 3090 TI" },
        { pattern: /\b3090\b/i, nvidia: "RTX 3090" },
        { pattern: /\b3080\s*TI\b/i, nvidia: "RTX 3080 TI" },
        { pattern: /\b3080\b/i, nvidia: "RTX 3080" },
        { pattern: /\b3070\s*TI\b/i, nvidia: "RTX 3070 TI" },
        { pattern: /\b3070\b/i, nvidia: "RTX 3070" },
        { pattern: /\b3060\s*TI\b/i, nvidia: "RTX 3060 TI" },
        { pattern: /\b3060\b/i, nvidia: "RTX 3060" },
        { pattern: /\b3050\b/i, nvidia: "RTX 3050" },

        // NVIDIA RTX 20 series
        { pattern: /\b2080\s*TI\b/i, nvidia: "RTX 2080 TI" },
        { pattern: /\b2080\s*SUPER\b/i, nvidia: "RTX 2080 SUPER" },
        { pattern: /\b2080\b/i, nvidia: "RTX 2080" },
        { pattern: /\b2070\s*SUPER\b/i, nvidia: "RTX 2070 SUPER" },
        { pattern: /\b2070\b/i, nvidia: "RTX 2070" },
        { pattern: /\b2060\s*SUPER\b/i, nvidia: "RTX 2060 SUPER" },
        { pattern: /\b2060\b/i, nvidia: "RTX 2060" },

        // NVIDIA GTX 16 series
        { pattern: /\b1660\s*TI\b/i, nvidia: "GTX 1660 TI" },
        { pattern: /\b1660\s*SUPER\b/i, nvidia: "GTX 1660 SUPER" },
        { pattern: /\b1660\b/i, nvidia: "GTX 1660" },
        { pattern: /\b1650\s*SUPER\b/i, nvidia: "GTX 1650 SUPER" },
        { pattern: /\b1650\b/i, nvidia: "GTX 1650" },
        { pattern: /\b1630\b/i, nvidia: "GTX 1630" },

        // NVIDIA GTX 10 series
        { pattern: /\b1080\s*TI\b/i, nvidia: "GTX 1080 TI" },
        { pattern: /\b1080\b/i, nvidia: "GTX 1080" },
        { pattern: /\b1070\s*TI\b/i, nvidia: "GTX 1070 TI" },
        { pattern: /\b1070\b/i, nvidia: "GTX 1070" },
        { pattern: /\b1060\b/i, nvidia: "GTX 1060" },
        { pattern: /\b1050\s*TI\b/i, nvidia: "GTX 1050 TI" },
        { pattern: /\b1050\b/i, nvidia: "GTX 1050" },

        // NVIDIA GTX 900 series
        { pattern: /\b980\s*TI\b/i, nvidia: "GTX 980 TI" },
        { pattern: /\b980\b/i, nvidia: "GTX 980" },
        { pattern: /\b970\b/i, nvidia: "GTX 970" },
        { pattern: /\b960\b/i, nvidia: "GTX 960" },
        { pattern: /\b950\b/i, nvidia: "GTX 950" },

        // NVIDIA GTX 700 / 600 / 500 / 400 series
        { pattern: /\b780\s*TI\b/i, nvidia: "GTX 780 TI" },
        { pattern: /\b780\b/i, nvidia: "GTX 780" },
        { pattern: /\b770\b/i, nvidia: "GTX 770" },
        { pattern: /\b760\b/i, nvidia: "GTX 760" },
        { pattern: /\b750\s*TI\b/i, nvidia: "GTX 750 TI" },
        { pattern: /\b750\b/i, nvidia: "GTX 750" },
        { pattern: /\b680\b/i, nvidia: "GTX 680" },
        { pattern: /\b670\b/i, nvidia: "GTX 670" },
        { pattern: /\b660\s*TI\b/i, nvidia: "GTX 660 TI" },
        { pattern: /\b660\b/i, nvidia: "GTX 660" },
        { pattern: /\b650\s*TI\b/i, nvidia: "GTX 650 TI" },
        { pattern: /\b650\b/i, nvidia: "GTX 650" },
        { pattern: /\b580\b/i, nvidia: "GTX 580", amd: "RX 580" },
        { pattern: /\b570\b/i, nvidia: "GTX 570", amd: "RX 570" },
        { pattern: /\b560\s*TI\b/i, nvidia: "GTX 560 TI" },
        { pattern: /\b560\b/i, nvidia: "GTX 560", amd: "RX 560" },
        { pattern: /\b550\s*TI\b/i, nvidia: "GTX 550 TI" },
        { pattern: /\b460\b/i, nvidia: "GTX 460", amd: "RX 460" },

        // AMD RX 7000 series
        { pattern: /\b7900\s*XTX\b/i, amd: "RX 7900 XTX" },
        { pattern: /\b7900\s*XT\b/i, amd: "RX 7900 XT" },
        { pattern: /\b7900\s*GRE\b/i, amd: "RX 7900 GRE" },
        { pattern: /\b7800\s*XT\b/i, amd: "RX 7800 XT" },
        { pattern: /\b7700\s*XT\b/i, amd: "RX 7700 XT" },
        { pattern: /\b7600\s*XT\b/i, amd: "RX 7600 XT" },
        { pattern: /\b7600\b/i, amd: "RX 7600" },

        // AMD RX 6000 series
        { pattern: /\b6950\s*XT\b/i, amd: "RX 6950 XT" },
        { pattern: /\b6900\s*XT\b/i, amd: "RX 6900 XT" },
        { pattern: /\b6800\s*XT\b/i, amd: "RX 6800 XT" },
        { pattern: /\b6800\b/i, amd: "RX 6800" },
        { pattern: /\b6750\s*XT\b/i, amd: "RX 6750 XT" },
        { pattern: /\b6700\s*XT\b/i, amd: "RX 6700 XT" },
        { pattern: /\b6700\b/i, amd: "RX 6700" },
        { pattern: /\b6650\s*XT\b/i, amd: "RX 6650 XT" },
        { pattern: /\b6600\s*XT\b/i, amd: "RX 6600 XT" },
        { pattern: /\b6600\b/i, amd: "RX 6600" },
        { pattern: /\b6500\s*XT\b/i, amd: "RX 6500 XT" },
        { pattern: /\b6400\b/i, amd: "RX 6400" },

        // AMD RX 5000 series
        { pattern: /\b5700\s*XT\b/i, amd: "RX 5700 XT" },
        { pattern: /\b5700\b/i, amd: "RX 5700" },
        { pattern: /\b5600\s*XT\b/i, amd: "RX 5600 XT" },
        { pattern: /\b5500\s*XT\b/i, amd: "RX 5500 XT" },

        // AMD RX 500/400 series
        { pattern: /\b590\b/i, amd: "RX 590" },
        { pattern: /\b480\b/i, amd: "RX 480" },
        { pattern: /\b470\b/i, amd: "RX 470" },

        // Intel ARC
        { pattern: /\bA770\b/i, intel: "ARC A770" },
        { pattern: /\bA750\b/i, intel: "ARC A750" },
        { pattern: /\bA580\b/i, intel: "ARC A580" },
        { pattern: /\bA380\b/i, intel: "ARC A380" },
        { pattern: /\bA310\b/i, intel: "ARC A310" }
    ];

    // Canonical GPU specifications mapping: model -> { default_vram, valid_vrams, manufacturer }
    const GPU_SPECS_DATASET = {
  "GT 1010": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 1010 DDR4": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 1030": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0,
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 1030 DDR4": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 1030 GK 107": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 120 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 120 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 120 MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 120 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 130 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 130 MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 130 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 140 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 220": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 220 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 220 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 230": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 230 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 230 OEM": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 240": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 240 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 240 M LE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 320 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 320 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 325 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 330 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 330 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 330 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 335 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 340 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 415 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 415 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 420 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 420 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 425 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 430": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 430 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 430 PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 435 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 440": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 440 MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 440 OEM": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 445 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 520": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 520 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 520 MX": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 520 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 520 PCI": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 520 PCIE X1": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 525 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 530 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 540 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 545": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 545 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 550 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 555 M": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 610": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 610 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 610 PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 610 PCIE X1": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 620": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 620 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 620 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 625 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 625 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 630": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 630 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 630 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 630 REV. 2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 630 REV. 2 PCIE X8": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 635 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 635 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640 M LE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640 OEM REBRAND": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 640 REV. 2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 645 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 645 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 650 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 650 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 705 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 710": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 710 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 710 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 710 PCIE X1": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 720": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 720 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 720 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 720 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 730": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0,
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 730 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 730 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 730 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 735 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 740": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 740 A": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 740 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 740 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 745 A": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 745 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 750 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 750 M MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 755 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GT 755 M MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 150 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 150 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 160 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 240 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 250": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 250 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 260 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 350 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 360 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 450": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 450 OEM": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 450 REV. 2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTS 450 REV. 3": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1050": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0,
      3.0,
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1050 MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1050 MOBILE": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1050 TI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1050 TI MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1050 TI MOBILE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060": {
    "default_vram": 6.0,
    "valid_vrams": [
      3.0,
      5.0,
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 3 GB GP 104": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 6 GB 9GBPS": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 6 GB GDDR5X": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 6 GB GP 104": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 6 GB REV. 2": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 MAX Q": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1060 MOBILE": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1070": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1070 GDDR5X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1070 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1070 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1070 TI": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1080": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1080 11GBPS": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1080 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1080 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1080 TI": {
    "default_vram": 11.0,
    "valid_vrams": [
      8.0,
      11.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1630": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 GDDR6": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 SUPER": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 TI MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 TI MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 TU 106": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1650 TU 116": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1660": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1660 SUPER": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1660 TI": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1660 TI MAX Q": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 1660 TI MOBILE": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 260": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 260 CORE 216": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 260 CORE 216 REV. 2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 260 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 260 OEM": {
    "default_vram": 1.75,
    "valid_vrams": [
      1.75
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 260 REV. 2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 275": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 275 PHYSX EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 280": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 280 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 285": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 285 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 285 MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 285 X2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 295": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 295 SINGLE PCB": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 M": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 SE": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 SE V2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 V2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 V2 ES": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 460 X2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 465": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 470": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 470 M": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 480": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 480 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 485 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 550": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 550 TI": {
    "default_vram": 5.0,
    "valid_vrams": [
      1.0,
      2.0,
      5.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 555 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 M": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 OEM": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 SE": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 TI": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 TI 448": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 TI OEM": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 560 TI X2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 570": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 570 M": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 570 REV. 2": {
    "default_vram": 1.25,
    "valid_vrams": [
      1.25
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 580": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 580 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 580 REV. 2": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 590": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 645 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 650": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 650 TI": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 650 TI BOOST": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 650 TI OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 660": {
    "default_vram": 5.0,
    "valid_vrams": [
      0.5,
      1.5,
      2.0,
      5.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 660 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 660 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 660 OEM": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 660 REV. 2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 660 TI": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 670": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 670 M": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 670 MX": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 675 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 675 MX": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 675 MX MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 680": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 680 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 680 MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 680 MX MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 690": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 745": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 745 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 750": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 750 GM 206": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 750 TI": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760": {
    "default_vram": 2.0,
    "valid_vrams": [
      1.0,
      1.5,
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 OEM REBRAND": {
    "default_vram": 1.5,
    "valid_vrams": [
      1.5
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 TI OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 TI OEM REBRAND": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 760 X2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 765 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 770": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 770 M": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 775 M MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 780": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0,
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 780 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 780 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 780 REV. 2": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 780 TI": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 850 A": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 850 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 860 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 860 M OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 870 M": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 880 M": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 950": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0,
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 950 A": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 950 LOW POWER": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 950 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 950 M MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 950 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 960": {
    "default_vram": 4.0,
    "valid_vrams": [
      2.0,
      3.0,
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 960 A": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 960 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 960 OEM": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 965 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 970": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0,
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 970 M": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 980": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0,
      6.0,
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 980 M": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 980 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 980 MX": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX 980 TI": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX TITAN": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX TITAN BLACK": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX TITAN X": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "GTX TITAN Z": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "HD 2350 PRO": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2400": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2400 PRO": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2400 PRO AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2400 PRO PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2400 XT": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2600 PRO": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2600 PRO AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2600 XT": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2600 XT AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2600 XT MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2600 XT X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2900 GT": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2900 PRO": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 2900 XT": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3200 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3200 MOBILE IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3300 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3410": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3450": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3450 AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3450 PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3450 X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3470": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3550": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3570": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3610": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3650": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3650 AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3690": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3730": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3750": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3830": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3850": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3850 AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3850 X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3870": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3870 MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 3870 X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4200 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4250": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4250 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4290 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4350": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4350 AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4350 PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4350 PCIE X1": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4450": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4520": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4550": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4570": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4570 REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4580": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4650": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4650 AGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4670": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4670 AGP": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4670 X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4700": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4710": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4720": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4730": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4730 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4750": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4770": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4810": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4830": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4850": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4850 X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4855": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4860": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4870": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4870 MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4870 X2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 4890": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5450": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5450 PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5450 PCIE X1": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5470": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5490": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5530": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5550": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5570": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5570 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5630": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5670": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5670 640SP EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5690": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5730": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5750": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5770": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5770 MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5770 X2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5830": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5850": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5870": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5870 EYEFINITY 6": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5870 MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 5970": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6230": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6250": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6250 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6290": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6290 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6310 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6320 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6330 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6350": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6350 A": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6350 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6370 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6370 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6380 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6390": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6410 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6430 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6450": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6450 A": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6450 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6450 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6470 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6480 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6490": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6490 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6490 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6510": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6520 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6530": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6530 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6530 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6550 A": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6550 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6550 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6570": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6570 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6570 M MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6570 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6610": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6610 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6620 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6625 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6630 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6630 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6650 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6650 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6670": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6670 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6730 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6750": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6750 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6750 M MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6770": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6770 GREEN EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6770 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6770 M MAC EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6790": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6830 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6850": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6850 1440SP EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6850 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6850 X2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6870": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6870 1600SP EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6870 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6870 X2": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6930": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6950": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6950 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6970": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6970 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6970 M MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6970 M REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6970 M X2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6990": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6990 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 6990 M REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7290 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7310 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7330 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7340 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7350 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7350 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7350 OEM PCI": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7370 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7400 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7410 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7420 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7430 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7450 A": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7450 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7450 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7470 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7470 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7470 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7480 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7490 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7500 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7510 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7510 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7520 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7530 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7540 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7550 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7560 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7560 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7570": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7570 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7570 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7590 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7600 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7610 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7620 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7630 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7640 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7650 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7650 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7650 M REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7660 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7660 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7670 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7670 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7670 M REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7670 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7690 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7690 M REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7690 M XT": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7690 M XT REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7720 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7730": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7730 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7750": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7750 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7770 GHZ EDITION": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7770 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7790": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7850": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7850 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7870 GHZ EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7870 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7870 XT": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7950": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7950 BOOST": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7950 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7950 MAC EDITION": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7950 MONICA BIOS 1": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7950 MONICA BIOS 2": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7970": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7970 GHZ EDITION": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7970 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7970 M X2": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7970 X2": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 7990": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8180 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8210 E": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8210 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8240 MOBILE IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8250 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8280 E": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8280 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8280 MOBILE IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8310 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8330 E": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8330 MOBILE IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8350 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8350 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8370 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8400 E": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8400 IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8400 MOBILE IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8410 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8450 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8450 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8470 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8470 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8490 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8510 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8510 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8530 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8550 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8550 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8550 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8550 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8570 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8570 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8570 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8570 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8570 OEM REBRAND": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8590 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8610 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8650 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8650 G IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8670 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8670 D IGP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8670 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8670 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8690 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8730 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8730 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8730 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8740 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8750 A": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8750 M": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8760 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8770 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8770 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8790 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8830 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8850 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8860 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8870 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8870 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8950 M": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8950 OEM": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8970 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8970 OEM": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD 8990 OEM": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 10EU": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 10EU MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 12EU": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 12EU MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 16EU MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 2000": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 3000": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 3000 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 32EU MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 400 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4000": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4000 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 405 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4200 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4400": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4400 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4600": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 4600 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 500 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 5000 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 505 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 510": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 510 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 515 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 520 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 530": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 530 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 5300 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 5500 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 5600 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 6000 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 610": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 610 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 615 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 620 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 630": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 630 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 6EU": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS 6EU MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 3000": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 4000": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 4600": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 4700": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 530": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 5700": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "HD GRAPHICS P 630 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R 7 260": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "NVIDIA"
  },
  "R 9 380": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 240": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 240 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 250": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 250 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 250E": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 250X": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 250XE": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 260": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 260X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 265": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 265X OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 340 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 350": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 350 640SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 350 FAKE CARD": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 350 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 350X OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 360": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 360 896SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 360E": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 370": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 430 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 435 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 450 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 A 260": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 A 265": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 A 360": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 GRAPHICS": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 260": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 260 DX": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 260 X": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 265": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 265 DX": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 270": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 270 DX": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 340": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 350": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 360": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 365 X": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 370": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 380": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 440": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 445": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 460": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 465": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 M 465 X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R7 MOBILE GRAPHICS": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R7E MOBILE GRAPHICS": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 255 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 260 OEM": {
    "default_vram": 1.0,
    "valid_vrams": [
      1.0
    ],
    "manufacturer": "AMD"
  },
  "R9 270": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 270 1024SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 270X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 280": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "R9 280X": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "R9 285": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 290": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 290X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 290X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 295X2": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 360 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 370": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 370 1024SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 370X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 380": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 380 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 380X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 390": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "R9 390 X2": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "R9 390X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "R9 A 375": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 FURY": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 FURY X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 265 X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 270 X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 275": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 275 X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 280 X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 290 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 290 X MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 295 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 295 X MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 360": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 365 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 370 X MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 375": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 375 X": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 380": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 380 MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 385": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 385 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 390 MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 390 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 395 MAC EDITION": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 395 X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 395 X MAC EDITION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 470": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 470 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "R9 M 485 X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "R9 NANO": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RTX 1000 MOBILE ADA GENERATION": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2000 ADA GENERATION": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2000 EMBEDDED ADA GENERATION": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2000 MAX Q ADA GENERATION": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2000 MOBILE ADA GENERATION": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2050 MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2050 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0,
      8.0,
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 MAX Q": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 MAX Q REFRESH": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 MOBILE": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 MOBILE REFRESH": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 SUPER": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 SUPER MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2060 TU 104": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 MAX Q REFRESH": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 MOBILE REFRESH": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 SUPER": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 SUPER MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2070 SUPER MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080 SUPER": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080 SUPER MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080 SUPER MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 2080 TI": {
    "default_vram": 11.0,
    "valid_vrams": [
      11.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3000 MOBILE ADA GENERATION": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050": {
    "default_vram": 8.0,
    "valid_vrams": [
      4.0,
      6.0,
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 8 GB GA 107": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 A MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 MAX Q REFRESH": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 MOBILE REFRESH": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 OEM": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 TI MAX Q": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3050 TI MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060": {
    "default_vram": 12.0,
    "valid_vrams": [
      6.0,
      8.0,
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 12 GB GA 104": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 3840SP": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 8 GB GA 104": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 MAX Q": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 MOBILE": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 TI": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 TI GA 103": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3060 TI GDDR6X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 TI": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 TI 8 GB GA 102": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 TI MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 TI MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3070 TIM": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3080": {
    "default_vram": 10.0,
    "valid_vrams": [
      10.0,
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3080 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3080 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3080 TI": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0,
      20.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3080 TI MAX Q": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3080 TI MOBILE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3090": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3090 TI": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3500 EMBEDDED ADA GENERATION": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 3500 MOBILE ADA GENERATION": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4000 ADA GENERATION": {
    "default_vram": 20.0,
    "valid_vrams": [
      20.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4000 MOBILE ADA GENERATION": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4000 SFF ADA GENERATION": {
    "default_vram": 20.0,
    "valid_vrams": [
      20.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4010": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4050 MAX Q": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4050 MOBILE": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4060": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4060 AD 106": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4060 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4060 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4060 TI": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4060 TI AD 104": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 AD 103": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 GDDR6": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 SUPER": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 TI": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 TI SUPER": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4070 TI SUPER AD 102": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4080": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4080 MAX Q": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4080 MOBILE": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4080 SUPER": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4090": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4090 D": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4090 MAX Q": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4090 MOBILE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 4500 ADA GENERATION": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 500 MOBILE ADA GENERATION": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5000 ADA GENERATION": {
    "default_vram": 32.0,
    "valid_vrams": [
      32.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5000 EMBEDDED ADA GENERATION": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5000 EMBEDDED ADA GENERATION X2": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5000 MAX Q ADA GENERATION": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5000 MOBILE ADA GENERATION": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5050": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5050 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5060": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5060 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5060 TI": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5070": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5070 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5070 SUPER": {
    "default_vram": 18.0,
    "valid_vrams": [
      18.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5070 TI": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5070 TI MOBILE": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5070 TI SUPER": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5080": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5080 MOBILE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5080 SUPER": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5090": {
    "default_vram": 32.0,
    "valid_vrams": [
      32.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5090 D": {
    "default_vram": 32.0,
    "valid_vrams": [
      32.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5090 D V2": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5090 MOBILE": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 5880 ADA GENERATION": {
    "default_vram": 48.0,
    "valid_vrams": [
      48.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX 6000 ADA GENERATION": {
    "default_vram": 48.0,
    "valid_vrams": [
      48.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 1000": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 1000 EMBEDDED": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 1000 MOBILE": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 2000": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 2000 EMBEDDED": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 2000 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 2000 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 3000 MOBILE": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 400": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4000": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4000 H": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4000 MAX Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4000 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4500": {
    "default_vram": 20.0,
    "valid_vrams": [
      20.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4500 EMBEDDED": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4500 MAX Q": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 4500 MOBILE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 500": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 500 EMBEDDED": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 500 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5000": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5000 12Q": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5000 8Q": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5000 MAX Q": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5000 MOBILE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5500": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5500 MAX Q": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 5500 MOBILE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A 6000": {
    "default_vram": 48.0,
    "valid_vrams": [
      48.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX A4 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 2000 BLACKWELL": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 4000 BLACKWELL": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 4000 BLACKWELL SFF": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 4500 BLACKWELL": {
    "default_vram": 32.0,
    "valid_vrams": [
      32.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 5000 72 GB BLACKWELL": {
    "default_vram": 72.0,
    "valid_vrams": [
      72.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 5000 BLACKWELL": {
    "default_vram": 48.0,
    "valid_vrams": [
      48.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 6000 BLACKWELL": {
    "default_vram": 96.0,
    "valid_vrams": [
      96.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 6000 BLACKWELL MAX Q": {
    "default_vram": 96.0,
    "valid_vrams": [
      96.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 6000 BLACKWELL SERVER": {
    "default_vram": 96.0,
    "valid_vrams": [
      96.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 6000 D BLACKWELL": {
    "default_vram": 96.0,
    "valid_vrams": [
      96.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RTX PRO 6000 D BLACKWELL MAX Q": {
    "default_vram": 96.0,
    "valid_vrams": [
      96.0
    ],
    "manufacturer": "NVIDIA"
  },
  "RX 440": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 455 OEM": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 460": {
    "default_vram": 4.0,
    "valid_vrams": [
      2.0,
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 460 1024SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 460 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 470": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0,
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 470 D": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 470 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 480": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 480 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5300 M": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5300 OEM": {
    "default_vram": 3.0,
    "valid_vrams": [
      3.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5300 XT OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 540 MOBILE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 540 X MOBILE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0,
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550 512SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550 640SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550 MOBILE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550 X 640SP": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 550 X MOBILE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5500 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5500 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5500 XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      4.0,
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0,
      6.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 896SP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 D": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 DX": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 X": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 X MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 560 XT": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5600 M": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5600 OEM": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5600 XT": {
    "default_vram": 6.0,
    "valid_vrams": [
      6.0
    ],
    "manufacturer": "AMD"
  },
  "RX 570": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0,
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 570 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 570 X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5700": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5700 M": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5700 XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 5700 XT 50TH ANNIVERSARY": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580": {
    "default_vram": 8.0,
    "valid_vrams": [
      4.0,
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580 2048SP": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580 G": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580 OEM": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580 X": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 580 X MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 590": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 590 GME": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 640 MOBILE": {
    "default_vram": 2.0,
    "valid_vrams": [
      2.0
    ],
    "manufacturer": "AMD"
  },
  "RX 640 OEM": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6400": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6450 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6500 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6500 XT": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6550 M": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6550 S": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6600": {
    "default_vram": 8.0,
    "valid_vrams": [
      4.0,
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6600 LE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6600 M": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6600 S": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6600 XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6650 M": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6650 M XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6650 XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6700": {
    "default_vram": 10.0,
    "valid_vrams": [
      10.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6700 M": {
    "default_vram": 10.0,
    "valid_vrams": [
      10.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6700 S": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6700 XT": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6750 GRE": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6750 XT": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6800": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6800 M": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6800 S": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6800 XT": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6850 M XT": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6900 XT": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 6950 XT": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7400": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7600": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7600 M": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7600 M XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7600 S": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7600 XT": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7650 GRE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7700": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7700 S": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7700 XT": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7800 M": {
    "default_vram": 12.0,
    "valid_vrams": [
      12.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7800 XT": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7900 GRE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7900 M": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7900 XT": {
    "default_vram": 20.0,
    "valid_vrams": [
      20.0
    ],
    "manufacturer": "AMD"
  },
  "RX 7900 XTX": {
    "default_vram": 24.0,
    "valid_vrams": [
      24.0
    ],
    "manufacturer": "AMD"
  },
  "RX 9060": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 9060 XT": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX 9070": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 9070 GRE": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX 9070 XT": {
    "default_vram": 16.0,
    "valid_vrams": [
      16.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 10 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 11": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 11 EMBEDDED": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 11 MOBILE": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 56": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 56 MOBILE": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 64": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 64 LIMITED EDITION": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA 64 LIQUID COOLING": {
    "default_vram": 8.0,
    "valid_vrams": [
      8.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA M GH": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  },
  "RX VEGA M GL": {
    "default_vram": 4.0,
    "valid_vrams": [
      4.0
    ],
    "manufacturer": "AMD"
  }
};

    // Canonical models sorted by length descending so longer tokens match first (e.g. RTX 4070 TI SUPER before RTX 4070)
    const CANONICAL_MODELS = [
  "RTX 5000 EMBEDDED ADA GENERATION X2",
  "RTX 2000 EMBEDDED ADA GENERATION",
  "RTX 3500 EMBEDDED ADA GENERATION",
  "RTX 5000 EMBEDDED ADA GENERATION",
  "RTX 1000 MOBILE ADA GENERATION",
  "RTX 2000 MOBILE ADA GENERATION",
  "RTX 3000 MOBILE ADA GENERATION",
  "RTX 3500 MOBILE ADA GENERATION",
  "RTX 4000 MOBILE ADA GENERATION",
  "RTX 5000 MOBILE ADA GENERATION",
  "RTX PRO 6000 D BLACKWELL MAX Q",
  "RTX 2000 MAX Q ADA GENERATION",
  "RTX 500 MOBILE ADA GENERATION",
  "RTX 5000 MAX Q ADA GENERATION",
  "RTX PRO 6000 BLACKWELL SERVER",
  "RTX PRO 5000 72 GB BLACKWELL",
  "RTX PRO 6000 BLACKWELL MAX Q",
  "RTX 4000 SFF ADA GENERATION",
  "RX 5700 XT 50TH ANNIVERSARY",
  "RTX PRO 4000 BLACKWELL SFF",
  "RX VEGA 64 LIMITED EDITION",
  "RX VEGA 64 LIQUID COOLING",
  "HD GRAPHICS P 630 MOBILE",
  "RTX 4070 TI SUPER AD 102",
  "RTX PRO 6000 D BLACKWELL",
  "GTX 260 CORE 216 REV. 2",
  "HD GRAPHICS 10EU MOBILE",
  "HD GRAPHICS 12EU MOBILE",
  "HD GRAPHICS 16EU MOBILE",
  "HD GRAPHICS 3000 MOBILE",
  "HD GRAPHICS 32EU MOBILE",
  "HD GRAPHICS 4000 MOBILE",
  "HD GRAPHICS 4200 MOBILE",
  "HD GRAPHICS 4400 MOBILE",
  "HD GRAPHICS 4600 MOBILE",
  "HD GRAPHICS 5000 MOBILE",
  "HD GRAPHICS 5300 MOBILE",
  "HD GRAPHICS 5500 MOBILE",
  "HD GRAPHICS 5600 MOBILE",
  "HD GRAPHICS 6000 MOBILE",
  "RTX 2000 ADA GENERATION",
  "RTX 2060 MOBILE REFRESH",
  "RTX 2070 MOBILE REFRESH",
  "RTX 3050 MOBILE REFRESH",
  "RTX 3070 TI 8 GB GA 102",
  "RTX 4000 ADA GENERATION",
  "RTX 4500 ADA GENERATION",
  "RTX 5000 ADA GENERATION",
  "RTX 5880 ADA GENERATION",
  "RTX 6000 ADA GENERATION",
  "GTX 675 MX MAC EDITION",
  "GTX 680 MX MAC EDITION",
  "GTX 760 TI OEM REBRAND",
  "HD 2600 XT MAC EDITION",
  "HD 6850 1440SP EDITION",
  "HD 6870 1600SP EDITION",
  "HD GRAPHICS 400 MOBILE",
  "HD GRAPHICS 405 MOBILE",
  "HD GRAPHICS 500 MOBILE",
  "HD GRAPHICS 505 MOBILE",
  "HD GRAPHICS 510 MOBILE",
  "HD GRAPHICS 515 MOBILE",
  "HD GRAPHICS 520 MOBILE",
  "HD GRAPHICS 530 MOBILE",
  "HD GRAPHICS 610 MOBILE",
  "HD GRAPHICS 615 MOBILE",
  "HD GRAPHICS 620 MOBILE",
  "HD GRAPHICS 630 MOBILE",
  "HD GRAPHICS 6EU MOBILE",
  "R9 M 290 X MAC EDITION",
  "R9 M 295 X MAC EDITION",
  "R9 M 370 X MAC EDITION",
  "R9 M 395 X MAC EDITION",
  "RTX 2060 MAX Q REFRESH",
  "RTX 2070 MAX Q REFRESH",
  "RTX 3050 MAX Q REFRESH",
  "RTX PRO 2000 BLACKWELL",
  "RTX PRO 4000 BLACKWELL",
  "RTX PRO 4500 BLACKWELL",
  "RTX PRO 5000 BLACKWELL",
  "RTX PRO 6000 BLACKWELL",
  "GT 630 REV. 2 PCIE X8",
  "GTX 275 PHYSX EDITION",
  "GTX 660 M MAC EDITION",
  "GTX 775 M MAC EDITION",
  "GTX 780 M MAC EDITION",
  "GTX 950 M MAC EDITION",
  "HD 5670 640SP EDITION",
  "HD 6490 M MAC EDITION",
  "HD 6570 M MAC EDITION",
  "HD 6630 M MAC EDITION",
  "HD 6750 M MAC EDITION",
  "HD 6770 GREEN EDITION",
  "HD 6770 M MAC EDITION",
  "HD 6970 M MAC EDITION",
  "HD 7950 MONICA BIOS 1",
  "HD 7950 MONICA BIOS 2",
  "RTX 2060 SUPER MOBILE",
  "RTX 2070 SUPER MOBILE",
  "RTX 2080 SUPER MOBILE",
  "RTX 3060 12 GB GA 104",
  "GT 120 M MAC EDITION",
  "GT 330 M MAC EDITION",
  "GT 640 M MAC EDITION",
  "GT 650 M MAC EDITION",
  "GT 750 M MAC EDITION",
  "GT 755 M MAC EDITION",
  "GTX 1060 3 GB GP 104",
  "GTX 1060 6 GB GDDR5X",
  "GTX 1060 6 GB GP 104",
  "GTX 1060 6 GB REV. 2",
  "HD 7690 M XT REBRAND",
  "R9 M 380 MAC EDITION",
  "R9 M 390 MAC EDITION",
  "R9 M 395 MAC EDITION",
  "RTX 2070 SUPER MAX Q",
  "RTX 2080 SUPER MAX Q",
  "RTX 3050 8 GB GA 107",
  "RTX 3060 8 GB GA 104",
  "GTX 1060 6 GB 9GBPS",
  "GTX 285 MAC EDITION",
  "GTX 680 MAC EDITION",
  "GTX 760 OEM REBRAND",
  "HD 3870 MAC EDITION",
  "HD 4870 MAC EDITION",
  "HD 5770 MAC EDITION",
  "HD 5870 EYEFINITY 6",
  "HD 5870 MAC EDITION",
  "HD 7770 GHZ EDITION",
  "HD 7870 GHZ EDITION",
  "HD 7950 MAC EDITION",
  "HD 7970 GHZ EDITION",
  "HD 8570 OEM REBRAND",
  "R7E MOBILE GRAPHICS",
  "RTX A 1000 EMBEDDED",
  "RTX A 2000 EMBEDDED",
  "RTX A 4500 EMBEDDED",
  "RX VEGA 11 EMBEDDED",
  "GT 120 MAC EDITION",
  "GT 130 MAC EDITION",
  "GT 440 MAC EDITION",
  "GT 640 OEM REBRAND",
  "GTX 1050 TI MOBILE",
  "GTX 1650 TI MOBILE",
  "GTX 1660 TI MOBILE",
  "GTX 295 SINGLE PCB",
  "HD 3200 MOBILE IGP",
  "HD 8240 MOBILE IGP",
  "HD 8280 MOBILE IGP",
  "HD 8330 MOBILE IGP",
  "HD 8400 MOBILE IGP",
  "HD GRAPHICS P 3000",
  "HD GRAPHICS P 4000",
  "HD GRAPHICS P 4600",
  "HD GRAPHICS P 4700",
  "HD GRAPHICS P 5700",
  "R7 MOBILE GRAPHICS",
  "RTX 3050 TI MOBILE",
  "RTX 3060 TI GA 103",
  "RTX 3060 TI GDDR6X",
  "RTX 3070 TI MOBILE",
  "RTX 3080 TI MOBILE",
  "RTX 4060 TI AD 104",
  "RTX 5070 TI MOBILE",
  "RTX A 500 EMBEDDED",
  "GTX 1050 TI MAX Q",
  "GTX 1650 TI MAX Q",
  "GTX 1660 TI MAX Q",
  "GTX 950 LOW POWER",
  "HD 6970 M REBRAND",
  "HD 6990 M REBRAND",
  "HD 7650 M REBRAND",
  "HD 7670 M REBRAND",
  "HD 7690 M REBRAND",
  "HD GRAPHICS P 530",
  "RTX 3050 A MOBILE",
  "RTX 3050 TI MAX Q",
  "RTX 3070 TI MAX Q",
  "RTX 3080 TI MAX Q",
  "RTX 4070 TI SUPER",
  "RTX 5070 TI SUPER",
  "RTX A 1000 MOBILE",
  "RTX A 2000 MOBILE",
  "RTX A 3000 MOBILE",
  "RTX A 4000 MOBILE",
  "RTX A 4500 MOBILE",
  "RTX A 5000 MOBILE",
  "RTX A 5500 MOBILE",
  "RX VEGA 10 MOBILE",
  "RX VEGA 11 MOBILE",
  "RX VEGA 56 MOBILE",
  "GTX 260 CORE 216",
  "GTX 650 TI BOOST",
  "HD GRAPHICS 10EU",
  "HD GRAPHICS 12EU",
  "HD GRAPHICS 2000",
  "HD GRAPHICS 3000",
  "HD GRAPHICS 4000",
  "HD GRAPHICS 4400",
  "HD GRAPHICS 4600",
  "R7 350 FAKE CARD",
  "RTX A 2000 MAX Q",
  "RTX A 4000 MAX Q",
  "RTX A 4500 MAX Q",
  "RTX A 500 MOBILE",
  "RTX A 5000 MAX Q",
  "RTX A 5500 MAX Q",
  "GTX 1050 MOBILE",
  "GTX 1060 MOBILE",
  "GTX 1070 GDDR5X",
  "GTX 1070 MOBILE",
  "GTX 1080 11GBPS",
  "GTX 1080 MOBILE",
  "GTX 1650 MOBILE",
  "GTX 1650 TU 106",
  "GTX 1650 TU 116",
  "GTX TITAN BLACK",
  "HD 2400 PRO AGP",
  "HD 2400 PRO PCI",
  "HD 2600 PRO AGP",
  "HD 4350 PCIE X1",
  "HD 4570 REBRAND",
  "HD 5450 PCIE X1",
  "HD 7350 OEM PCI",
  "HD GRAPHICS 510",
  "HD GRAPHICS 530",
  "HD GRAPHICS 610",
  "HD GRAPHICS 630",
  "HD GRAPHICS 6EU",
  "RTX 2050 MOBILE",
  "RTX 2060 MOBILE",
  "RTX 2060 TU 104",
  "RTX 2070 MOBILE",
  "RTX 2080 MOBILE",
  "RTX 3050 MOBILE",
  "RTX 3060 3840SP",
  "RTX 3060 MOBILE",
  "RTX 3070 MOBILE",
  "RTX 3080 MOBILE",
  "RTX 4050 MOBILE",
  "RTX 4060 AD 106",
  "RTX 4060 MOBILE",
  "RTX 4070 AD 103",
  "RTX 4070 MOBILE",
  "RTX 4080 MOBILE",
  "RTX 4090 MOBILE",
  "RTX 5050 MOBILE",
  "RTX 5060 MOBILE",
  "RTX 5070 MOBILE",
  "RTX 5080 MOBILE",
  "RTX 5090 MOBILE",
  "RX 540 X MOBILE",
  "RX 550 X MOBILE",
  "RX 560 X MOBILE",
  "RX 580 X MOBILE",
  "GT 1030 GK 107",
  "GT 520 PCIE X1",
  "GT 610 PCIE X1",
  "GT 710 PCIE X1",
  "GTS 450 REV. 2",
  "GTS 450 REV. 3",
  "GTX 1050 MAX Q",
  "GTX 1060 MAX Q",
  "GTX 1070 MAX Q",
  "GTX 1080 MAX Q",
  "GTX 1650 GDDR6",
  "GTX 1650 MAX Q",
  "GTX 1650 SUPER",
  "GTX 1660 SUPER",
  "GTX 260 REV. 2",
  "GTX 560 TI 448",
  "GTX 560 TI OEM",
  "GTX 570 REV. 2",
  "GTX 580 REV. 2",
  "GTX 650 TI OEM",
  "GTX 660 REV. 2",
  "GTX 750 GM 206",
  "GTX 760 TI OEM",
  "GTX 780 REV. 2",
  "GTX 980 MOBILE",
  "HD 2600 XT AGP",
  "RTX 2050 MAX Q",
  "RTX 2060 MAX Q",
  "RTX 2060 SUPER",
  "RTX 2070 MAX Q",
  "RTX 2070 SUPER",
  "RTX 2080 MAX Q",
  "RTX 2080 SUPER",
  "RTX 3050 MAX Q",
  "RTX 3060 MAX Q",
  "RTX 3070 MAX Q",
  "RTX 3080 MAX Q",
  "RTX 4050 MAX Q",
  "RTX 4060 MAX Q",
  "RTX 4070 GDDR6",
  "RTX 4070 MAX Q",
  "RTX 4070 SUPER",
  "RTX 4080 MAX Q",
  "RTX 4080 SUPER",
  "RTX 4090 MAX Q",
  "RTX 5070 SUPER",
  "RTX 5080 SUPER",
  "RTX A 5000 12Q",
  "RX 5300 XT OEM",
  "RX 550 X 640SP",
  "GT 630 REV. 2",
  "GT 640 REV. 2",
  "GTX 460 SE V2",
  "GTX 460 V2 ES",
  "GTX 560 TI X2",
  "GTX 860 M OEM",
  "HD 2600 XT X2",
  "HD 6370 D IGP",
  "HD 6380 G IGP",
  "HD 6410 D IGP",
  "HD 6480 G IGP",
  "HD 6520 G IGP",
  "HD 6530 D IGP",
  "HD 6550 D IGP",
  "HD 6620 G IGP",
  "HD 7400 G IGP",
  "HD 7420 G IGP",
  "HD 7480 D IGP",
  "HD 7500 G IGP",
  "HD 7520 G IGP",
  "HD 7540 D IGP",
  "HD 7560 D IGP",
  "HD 7560 G IGP",
  "HD 7600 G IGP",
  "HD 7620 G IGP",
  "HD 7640 G IGP",
  "HD 7660 D IGP",
  "HD 7660 G IGP",
  "HD 7950 BOOST",
  "HD 8310 G IGP",
  "HD 8350 G IGP",
  "HD 8370 D IGP",
  "HD 8410 G IGP",
  "HD 8450 G IGP",
  "HD 8470 D IGP",
  "HD 8510 G IGP",
  "HD 8550 D IGP",
  "HD 8550 G IGP",
  "HD 8570 D IGP",
  "HD 8610 G IGP",
  "HD 8650 D IGP",
  "HD 8650 G IGP",
  "HD 8670 D IGP",
  "R9 270 1024SP",
  "R9 370 1024SP",
  "RTX 5090 D V2",
  "RTX A 5000 8Q",
  "RTX A4 MOBILE",
  "RX 460 1024SP",
  "RX 460 MOBILE",
  "RX 470 MOBILE",
  "RX 480 MOBILE",
  "RX 540 MOBILE",
  "RX 550 MOBILE",
  "RX 560 MOBILE",
  "RX 570 MOBILE",
  "RX 580 2048SP",
  "RX 580 MOBILE",
  "RX 640 MOBILE",
  "GT 1010 DDR4",
  "GT 1030 DDR4",
  "HD 6970 M X2",
  "HD 7690 M XT",
  "HD 7970 M X2",
  "R7 350 640SP",
  "R7 360 896SP",
  "RTX 3050 OEM",
  "RTX 3070 TIM",
  "RTX A 4000 H",
  "RX 550 512SP",
  "RX 550 640SP",
  "RX 560 896SP",
  "RX 6650 M XT",
  "RX 6850 M XT",
  "RX 7600 M XT",
  "RX VEGA M GH",
  "RX VEGA M GL",
  "GT 240 M LE",
  "GT 640 M LE",
  "GTS 150 OEM",
  "GTS 240 OEM",
  "GTS 450 OEM",
  "GTX 1050 TI",
  "GTX 1070 TI",
  "GTX 1080 TI",
  "GTX 1660 TI",
  "GTX 260 OEM",
  "GTX 460 OEM",
  "GTX 555 OEM",
  "GTX 560 OEM",
  "GTX 645 OEM",
  "GTX 660 OEM",
  "GTX 745 OEM",
  "GTX 760 OEM",
  "GTX 950 OEM",
  "GTX 960 OEM",
  "GTX TITAN X",
  "GTX TITAN Z",
  "HD 2350 PRO",
  "HD 2400 PRO",
  "HD 2600 PRO",
  "HD 2900 PRO",
  "HD 3200 IGP",
  "HD 3300 IGP",
  "HD 3450 AGP",
  "HD 3450 PCI",
  "HD 3650 AGP",
  "HD 3850 AGP",
  "HD 4200 IGP",
  "HD 4250 IGP",
  "HD 4290 IGP",
  "HD 4350 AGP",
  "HD 4350 PCI",
  "HD 4650 AGP",
  "HD 4670 AGP",
  "HD 4730 OEM",
  "HD 5450 PCI",
  "HD 5570 OEM",
  "HD 6250 IGP",
  "HD 6290 IGP",
  "HD 6310 IGP",
  "HD 6320 IGP",
  "HD 6450 OEM",
  "HD 6570 OEM",
  "HD 7290 IGP",
  "HD 7310 IGP",
  "HD 7340 IGP",
  "HD 7350 OEM",
  "HD 7450 OEM",
  "HD 7470 OEM",
  "HD 7510 OEM",
  "HD 7570 OEM",
  "HD 7670 OEM",
  "HD 7720 OEM",
  "HD 8180 IGP",
  "HD 8210 IGP",
  "HD 8250 IGP",
  "HD 8280 IGP",
  "HD 8350 OEM",
  "HD 8400 IGP",
  "HD 8450 OEM",
  "HD 8470 OEM",
  "HD 8490 OEM",
  "HD 8510 OEM",
  "HD 8550 OEM",
  "HD 8570 OEM",
  "HD 8670 OEM",
  "HD 8730 OEM",
  "HD 8740 OEM",
  "HD 8760 OEM",
  "HD 8770 OEM",
  "HD 8860 OEM",
  "HD 8870 OEM",
  "HD 8950 OEM",
  "HD 8970 OEM",
  "HD 8990 OEM",
  "R7 265X OEM",
  "R7 350X OEM",
  "R7 GRAPHICS",
  "R7 M 260 DX",
  "R7 M 265 DX",
  "R7 M 270 DX",
  "RTX 2080 TI",
  "RTX 3060 TI",
  "RTX 3070 TI",
  "RTX 3080 TI",
  "RTX 3090 TI",
  "RTX 4060 TI",
  "RTX 4070 TI",
  "RTX 5060 TI",
  "RTX 5070 TI",
  "RX 5300 OEM",
  "RX 5500 OEM",
  "RX 5600 OEM",
  "RX 6750 GRE",
  "RX 7650 GRE",
  "RX 7900 GRE",
  "RX 7900 XTX",
  "RX 9070 GRE",
  "GT 120 OEM",
  "GT 130 OEM",
  "GT 140 OEM",
  "GT 220 OEM",
  "GT 230 OEM",
  "GT 320 OEM",
  "GT 330 OEM",
  "GT 340 OEM",
  "GT 415 OEM",
  "GT 420 OEM",
  "GT 430 OEM",
  "GT 430 PCI",
  "GT 440 OEM",
  "GT 520 OEM",
  "GT 520 PCI",
  "GT 530 OEM",
  "GT 545 OEM",
  "GT 610 OEM",
  "GT 610 PCI",
  "GT 620 OEM",
  "GT 625 OEM",
  "GT 630 OEM",
  "GT 635 OEM",
  "GT 640 OEM",
  "GT 645 OEM",
  "GT 705 OEM",
  "GT 710 OEM",
  "GT 720 OEM",
  "GT 730 OEM",
  "GT 740 OEM",
  "GTX 285 X2",
  "GTX 460 SE",
  "GTX 460 V2",
  "GTX 460 X2",
  "GTX 550 TI",
  "GTX 560 SE",
  "GTX 560 TI",
  "GTX 650 TI",
  "GTX 660 TI",
  "GTX 670 MX",
  "GTX 675 MX",
  "GTX 750 TI",
  "GTX 760 X2",
  "GTX 780 TI",
  "GTX 980 MX",
  "GTX 980 TI",
  "HD 2400 XT",
  "HD 2600 XT",
  "HD 2900 GT",
  "HD 2900 XT",
  "HD 3450 X2",
  "HD 3850 X2",
  "HD 3870 X2",
  "HD 4670 X2",
  "HD 4850 X2",
  "HD 4870 X2",
  "HD 5770 X2",
  "HD 6850 X2",
  "HD 6870 X2",
  "HD 7870 XT",
  "HD 7970 X2",
  "R7 240 OEM",
  "R7 250 OEM",
  "R7 340 OEM",
  "R7 350 OEM",
  "R7 430 OEM",
  "R7 435 OEM",
  "R7 450 OEM",
  "R7 M 260 X",
  "R7 M 365 X",
  "R7 M 465 X",
  "R9 255 OEM",
  "R9 260 OEM",
  "R9 360 OEM",
  "R9 380 OEM",
  "R9 M 265 X",
  "R9 M 270 X",
  "R9 M 275 X",
  "R9 M 280 X",
  "R9 M 290 X",
  "R9 M 295 X",
  "R9 M 365 X",
  "R9 M 375 X",
  "R9 M 385 X",
  "R9 M 390 X",
  "R9 M 395 X",
  "R9 M 470 X",
  "R9 M 485 X",
  "RTX 4090 D",
  "RTX 5090 D",
  "RTX A 1000",
  "RTX A 2000",
  "RTX A 4000",
  "RTX A 4500",
  "RTX A 5000",
  "RTX A 5500",
  "RTX A 6000",
  "RX 455 OEM",
  "RX 5500 XT",
  "RX 5600 XT",
  "RX 5700 XT",
  "RX 580 OEM",
  "RX 590 GME",
  "RX 640 OEM",
  "RX 6500 XT",
  "RX 6600 LE",
  "RX 6600 XT",
  "RX 6650 XT",
  "RX 6700 XT",
  "RX 6750 XT",
  "RX 6800 XT",
  "RX 6900 XT",
  "RX 6950 XT",
  "RX 7600 XT",
  "RX 7700 XT",
  "RX 7800 XT",
  "RX 7900 XT",
  "RX 9060 XT",
  "RX 9070 XT",
  "RX VEGA 11",
  "RX VEGA 56",
  "RX VEGA 64",
  "GT 520 MX",
  "GTS 150 M",
  "GTS 160 M",
  "GTS 250 M",
  "GTS 260 M",
  "GTS 350 M",
  "GTS 360 M",
  "GTX 260 M",
  "GTX 280 M",
  "GTX 285 M",
  "GTX 460 M",
  "GTX 470 M",
  "GTX 480 M",
  "GTX 485 M",
  "GTX 560 M",
  "GTX 570 M",
  "GTX 580 M",
  "GTX 660 M",
  "GTX 670 M",
  "GTX 675 M",
  "GTX 680 M",
  "GTX 760 A",
  "GTX 760 M",
  "GTX 765 M",
  "GTX 770 M",
  "GTX 780 M",
  "GTX 850 A",
  "GTX 850 M",
  "GTX 860 M",
  "GTX 870 M",
  "GTX 880 M",
  "GTX 950 A",
  "GTX 950 M",
  "GTX 960 A",
  "GTX 960 M",
  "GTX 965 M",
  "GTX 970 M",
  "GTX 980 M",
  "GTX TITAN",
  "HD 6330 M",
  "HD 6350 A",
  "HD 6350 M",
  "HD 6370 M",
  "HD 6430 M",
  "HD 6450 A",
  "HD 6450 M",
  "HD 6470 M",
  "HD 6490 M",
  "HD 6530 M",
  "HD 6550 A",
  "HD 6550 M",
  "HD 6570 M",
  "HD 6610 M",
  "HD 6625 M",
  "HD 6630 M",
  "HD 6650 A",
  "HD 6650 M",
  "HD 6670 A",
  "HD 6730 M",
  "HD 6750 M",
  "HD 6770 M",
  "HD 6830 M",
  "HD 6850 M",
  "HD 6870 M",
  "HD 6950 M",
  "HD 6970 M",
  "HD 6990 M",
  "HD 7330 M",
  "HD 7350 M",
  "HD 7370 M",
  "HD 7410 M",
  "HD 7430 M",
  "HD 7450 A",
  "HD 7450 M",
  "HD 7470 A",
  "HD 7470 M",
  "HD 7490 M",
  "HD 7510 M",
  "HD 7530 M",
  "HD 7550 M",
  "HD 7570 M",
  "HD 7590 M",
  "HD 7610 M",
  "HD 7630 M",
  "HD 7650 A",
  "HD 7650 M",
  "HD 7670 A",
  "HD 7670 M",
  "HD 7690 M",
  "HD 7730 M",
  "HD 7750 M",
  "HD 7770 M",
  "HD 7850 M",
  "HD 7870 M",
  "HD 7950 M",
  "HD 7970 M",
  "HD 8210 E",
  "HD 8280 E",
  "HD 8330 E",
  "HD 8400 E",
  "HD 8530 M",
  "HD 8550 M",
  "HD 8570 A",
  "HD 8570 M",
  "HD 8590 M",
  "HD 8670 A",
  "HD 8670 M",
  "HD 8690 M",
  "HD 8730 A",
  "HD 8730 M",
  "HD 8750 A",
  "HD 8750 M",
  "HD 8770 M",
  "HD 8790 M",
  "HD 8830 M",
  "HD 8850 M",
  "HD 8870 M",
  "HD 8950 M",
  "HD 8970 M",
  "R9 390 X2",
  "R9 FURY X",
  "RTX A 400",
  "RTX A 500",
  "RX 5300 M",
  "RX 5500 M",
  "RX 560 DX",
  "RX 560 XT",
  "RX 5600 M",
  "RX 5700 M",
  "RX 6450 M",
  "RX 6500 M",
  "RX 6550 M",
  "RX 6550 S",
  "RX 6600 M",
  "RX 6600 S",
  "RX 6650 M",
  "RX 6700 M",
  "RX 6700 S",
  "RX 6800 M",
  "RX 6800 S",
  "RX 7600 M",
  "RX 7600 S",
  "RX 7700 S",
  "RX 7800 M",
  "RX 7900 M",
  "GT 120 M",
  "GT 130 M",
  "GT 220 M",
  "GT 230 M",
  "GT 240 M",
  "GT 320 M",
  "GT 325 M",
  "GT 330 M",
  "GT 335 M",
  "GT 415 M",
  "GT 420 M",
  "GT 425 M",
  "GT 435 M",
  "GT 445 M",
  "GT 520 M",
  "GT 525 M",
  "GT 540 M",
  "GT 550 M",
  "GT 555 M",
  "GT 620 M",
  "GT 625 M",
  "GT 630 M",
  "GT 635 M",
  "GT 640 M",
  "GT 645 M",
  "GT 650 M",
  "GT 710 M",
  "GT 720 A",
  "GT 720 M",
  "GT 730 A",
  "GT 730 M",
  "GT 735 M",
  "GT 740 A",
  "GT 740 M",
  "GT 745 A",
  "GT 745 M",
  "GT 750 M",
  "GT 755 M",
  "GTX 1050",
  "GTX 1060",
  "GTX 1070",
  "GTX 1080",
  "GTX 1630",
  "GTX 1650",
  "GTX 1660",
  "R7 250XE",
  "R7 A 260",
  "R7 A 265",
  "R7 A 360",
  "R7 M 260",
  "R7 M 265",
  "R7 M 270",
  "R7 M 340",
  "R7 M 350",
  "R7 M 360",
  "R7 M 370",
  "R7 M 380",
  "R7 M 440",
  "R7 M 445",
  "R7 M 460",
  "R7 M 465",
  "R9 290X2",
  "R9 295X2",
  "R9 A 375",
  "R9 M 275",
  "R9 M 360",
  "R9 M 375",
  "R9 M 380",
  "R9 M 385",
  "R9 M 470",
  "RTX 2060",
  "RTX 2070",
  "RTX 2080",
  "RTX 3050",
  "RTX 3060",
  "RTX 3070",
  "RTX 3080",
  "RTX 3090",
  "RTX 4010",
  "RTX 4060",
  "RTX 4070",
  "RTX 4080",
  "RTX 4090",
  "RTX 5050",
  "RTX 5060",
  "RTX 5070",
  "RTX 5080",
  "RTX 5090",
  "RX 470 D",
  "RX 550 X",
  "RX 560 D",
  "RX 560 X",
  "RX 570 X",
  "RX 580 G",
  "RX 580 X",
  "GT 1010",
  "GT 1030",
  "GTS 250",
  "GTS 450",
  "GTX 260",
  "GTX 275",
  "GTX 280",
  "GTX 285",
  "GTX 295",
  "GTX 460",
  "GTX 465",
  "GTX 470",
  "GTX 480",
  "GTX 550",
  "GTX 560",
  "GTX 570",
  "GTX 580",
  "GTX 590",
  "GTX 650",
  "GTX 660",
  "GTX 670",
  "GTX 680",
  "GTX 690",
  "GTX 745",
  "GTX 750",
  "GTX 760",
  "GTX 770",
  "GTX 780",
  "GTX 950",
  "GTX 960",
  "GTX 970",
  "GTX 980",
  "HD 2400",
  "HD 3410",
  "HD 3450",
  "HD 3470",
  "HD 3550",
  "HD 3570",
  "HD 3610",
  "HD 3650",
  "HD 3690",
  "HD 3730",
  "HD 3750",
  "HD 3830",
  "HD 3850",
  "HD 3870",
  "HD 4250",
  "HD 4350",
  "HD 4450",
  "HD 4520",
  "HD 4550",
  "HD 4570",
  "HD 4580",
  "HD 4650",
  "HD 4670",
  "HD 4700",
  "HD 4710",
  "HD 4720",
  "HD 4730",
  "HD 4750",
  "HD 4770",
  "HD 4810",
  "HD 4830",
  "HD 4850",
  "HD 4855",
  "HD 4860",
  "HD 4870",
  "HD 4890",
  "HD 5450",
  "HD 5470",
  "HD 5490",
  "HD 5530",
  "HD 5550",
  "HD 5570",
  "HD 5630",
  "HD 5670",
  "HD 5690",
  "HD 5730",
  "HD 5750",
  "HD 5770",
  "HD 5830",
  "HD 5850",
  "HD 5870",
  "HD 5970",
  "HD 6230",
  "HD 6250",
  "HD 6290",
  "HD 6350",
  "HD 6390",
  "HD 6450",
  "HD 6490",
  "HD 6510",
  "HD 6530",
  "HD 6570",
  "HD 6610",
  "HD 6670",
  "HD 6750",
  "HD 6770",
  "HD 6790",
  "HD 6850",
  "HD 6870",
  "HD 6930",
  "HD 6950",
  "HD 6970",
  "HD 6990",
  "HD 7570",
  "HD 7730",
  "HD 7750",
  "HD 7790",
  "HD 7850",
  "HD 7950",
  "HD 7970",
  "HD 7990",
  "R 7 260",
  "R 9 380",
  "R7 250E",
  "R7 250X",
  "R7 260X",
  "R7 360E",
  "R9 270X",
  "R9 280X",
  "R9 290X",
  "R9 370X",
  "R9 380X",
  "R9 390X",
  "R9 FURY",
  "R9 NANO",
  "RX 5700",
  "RX 6400",
  "RX 6600",
  "RX 6700",
  "RX 6800",
  "RX 7400",
  "RX 7600",
  "RX 7700",
  "RX 9060",
  "RX 9070",
  "GT 220",
  "GT 230",
  "GT 240",
  "GT 430",
  "GT 440",
  "GT 520",
  "GT 545",
  "GT 610",
  "GT 620",
  "GT 630",
  "GT 640",
  "GT 710",
  "GT 720",
  "GT 730",
  "GT 740",
  "R7 240",
  "R7 250",
  "R7 260",
  "R7 265",
  "R7 350",
  "R7 360",
  "R7 370",
  "R9 270",
  "R9 280",
  "R9 285",
  "R9 290",
  "R9 370",
  "R9 380",
  "R9 390",
  "RX 440",
  "RX 460",
  "RX 470",
  "RX 480",
  "RX 550",
  "RX 560",
  "RX 570",
  "RX 580",
  "RX 590"
];

    function normalizeText(text) {
        if (!text) return "";
        let clean = String(text).toUpperCase().replace(/[-_/]/g, " ").replace(/\s+/g, " ").trim();
        clean = clean.replace(/\bGEFORCE\b/g, " ");
        clean = clean.replace(/\bRADEON\b/g, " ");
        clean = clean.replace(/\bNVIDIA\b/g, " ");
        clean = clean.replace(/\bAMD\b/g, " ");
        clean = clean.replace(/\bINTEL\s+ARC\b/g, "ARC");
        // Separate prefix from numbers e.g. RTX3060 -> RTX 3060, RX580 -> RX 580, 1050TI -> 1050 TI
        clean = clean.replace(/\b(RTX|GTX|RX|GT|GTS|ARC|HD|R9|R7)\s*(\d{3,4})(?=\s*(?:XTX|SUPER|TI|XT)?\b)/g, "$1 $2");
        clean = clean.replace(/\b(\d{3,4})\s*(TI|XT|XTX|SUPER)\b/g, "$1 $2");
        clean = clean.replace(/\s+/g, " ").trim();
        return clean;
    }

    function extractModelCandidates(combinedText, detectedBrand) {
        if (!combinedText) return [];
        const rawNormalized = normalizeText(combinedText);
        const padded = ` ${rawNormalized} `;

        // Step 1: Search for explicit canonical models (with prefix e.g. GTX 970, RTX 3060, RX 580)
        const matches = [];
        for (const model of CANONICAL_MODELS) {
            const normModel = normalizeText(model);
            const regex = new RegExp(`(?:^|\\s)${normModel.replace(/\s+/g, '\\s+')}(?=\\s|$)`, 'i');
            if (regex.test(padded)) {
                matches.push(model);
            }
        }

        // Deduplicate sub-matches (e.g. keep RTX 3060 TI over RTX 3060)
        const filteredMatches = matches.filter(candidate => {
            const normalizedCandidate = normalizeText(candidate);
            return !matches.some(other => other !== candidate && normalizeText(other).startsWith(`${normalizedCandidate} `));
        });

        if (filteredMatches.length > 0) {
            return filteredMatches;
        }

        // Step 2: Standalone model number resolution (e.g., 'Zotac 970 4GB' -> 'GTX 970', 'Sapphire 580' -> 'RX 580')
        const upper = rawNormalized.toUpperCase();
        for (const rule of STANDALONE_MODEL_RULES) {
            if (rule.pattern.test(upper)) {
                if (rule.intel) return [rule.intel];
                if (rule.nvidia && !rule.amd) return [rule.nvidia];
                if (rule.amd && !rule.nvidia) return [rule.amd];
                if (rule.nvidia && rule.amd) {
                    const brand = (detectedBrand || "").toUpperCase();
                    if (AMD_ONLY_BRANDS.includes(brand) || /\b(RADEON|AMD|NITRO|PULSE|RED DEVIL|SWFT|MERC)\b/i.test(upper)) {
                        return [rule.amd];
                    }
                    if (NVIDIA_ONLY_BRANDS.includes(brand) || /\b(GEFORCE|NVIDIA)\b/i.test(upper)) {
                        return [rule.nvidia];
                    }
                    // Default disambiguation for resale market: 580/570/590/480/470 default to AMD RX
                    if (/\b(580|570|590|480|470)\b/.test(upper)) {
                        return [rule.amd];
                    }
                    return [rule.nvidia];
                }
            }
        }

        return [];
    }

    function extractModel(combinedText, detectedBrand) {
        const matches = extractModelCandidates(combinedText, detectedBrand);
        return matches && matches.length === 1 ? matches[0] : null;
    }

    /**
     * Strict VRAM extraction and auto-fill:
     * 1. Inspects explicit spec attributes (key_values.vram, key_values.memory)
     * 2. Inspects listing title for explicit VRAM (e.g. '8GB', '8 GB GDDR6')
     * 3. Inspects listing description with strict VRAM-specific keyword binding (never loose raw text)
     * 4. If VRAM is omitted from listing, auto-fills with official standard default_vram for the model from dataset.
     */
    function extractVram(pageContext, matchedModel) {
        const { title = "", raw_text = "", key_values = {} } = (pageContext || {});
        let explicitVram = null;

        // Priority 1: Scraped explicit spec field (e.g. 'VRAM: 8GB', 'Memory: 4 GB')
        if (key_values) {
            for (const key of ['vram', 'memory', 'memory size', 'graphics memory', 'vram (gb)', 'vram_gb', 'memory_size']) {
                if (key_values[key]) {
                    const m = String(key_values[key]).match(/\b(\d{1,2})\s*(?:GB|GIGABYTE|G)\b/i);
                    if (m) {
                        const val = parseFloat(m[1]);
                        if (val >= 1 && val <= 48) {
                            explicitVram = val;
                            break;
                        }
                    }
                }
            }
        }

        // Priority 2: Match in title (strict 'GB' or 'G' before GDDR/VRAM, preventing '1 Year Warranty' or '1G' collisions)
        if (explicitVram === null && title) {
            const titleMatch = title.match(/\b(\d{1,2})\s*GB\b(?!\s*warranty)/i) ||
                               title.match(/\b(\d{1,2})\s*G(?=\s*(?:GDDR\d[X]?|VRAM|DDR\d))/i);
            if (titleMatch) {
                const val = parseFloat(titleMatch[1]);
                if (val >= 1 && val <= 48) {
                    explicitVram = val;
                }
            }
        }

        // Priority 3: Description search with strict VRAM keywords binding only
        if (explicitVram === null && raw_text) {
            const descMatch = raw_text.match(/\b(\d{1,2})\s*GB\s*(?:GDDR\d[X]?|VRAM|DDR\d|VIDEO\s*MEMORY|GRAPHICS\s*MEMORY)\b/i) ||
                              raw_text.match(/\b(?:VRAM|GRAPHICS\s*MEMORY|MEMORY\s*SIZE|VIDEO\s*MEMORY)\s*[:=]?\s*(\d{1,2})\s*GB\b/i);
            if (descMatch) {
                const val = parseFloat(descMatch[1]);
                if (val >= 1 && val <= 48) {
                    explicitVram = val;
                }
            }
        }

        const spec = matchedModel ? GPU_SPECS_DATASET[matchedModel] : null;

        // If explicit VRAM was found, validate against official specifications
        if (explicitVram !== null) {
            const isValid = spec ? (spec.valid_vrams.includes(explicitVram) || spec.valid_vrams.length === 0) : true;
            return {
                vram_gb: explicitVram,
                is_auto_filled: false,
                is_valid: isValid,
                spec: spec
            };
        }

        // If no VRAM was mentioned, auto-fill with the official standard VRAM from dataset
        if (spec && spec.default_vram) {
            return {
                vram_gb: spec.default_vram,
                is_auto_filled: true,
                is_valid: true,
                spec: spec
            };
        }

        return {
            vram_gb: null,
            is_auto_filled: false,
            is_valid: false,
            spec: spec
        };
    }

    function extractBrand(combinedText, rawTextElements) {
        if (!combinedText) return "Any";
        const upper = combinedText.toUpperCase();

        if (rawTextElements && rawTextElements.brand) {
            const explicit = rawTextElements.brand.toUpperCase();
            for (const b of KNOWN_BRANDS) {
                if (explicit.includes(b)) return b;
            }
        }

        for (const b of KNOWN_BRANDS) {
            const regex = new RegExp(`\\b${b}\\b`, 'i');
            if (regex.test(upper)) {
                return b;
            }
        }

        return "Any";
    }

    function extractManufacturer(model, brand) {
        if (model && GPU_SPECS_DATASET[model] && GPU_SPECS_DATASET[model].manufacturer) {
            return GPU_SPECS_DATASET[model].manufacturer;
        }
        if (!model) return "Any";
        const mUpper = model.toUpperCase();
        if (mUpper.startsWith("RTX") || mUpper.startsWith("GTX") || mUpper.startsWith("GT") || mUpper.startsWith("GTS")) return "NVIDIA";
        if (mUpper.startsWith("RX") || mUpper.startsWith("R9") || mUpper.startsWith("R7") || mUpper.startsWith("HD")) return "AMD";
        if (mUpper.startsWith("ARC")) return "Intel";
        if (brand === "NVIDIA" || brand === "AMD" || brand === "INTEL") return brand;
        return "Any";
    }

    function parse(pageContext) {
        const { title = "", price = null, raw_text = "", key_values = {} } = pageContext;
        const brand = extractBrand(`${title} ${key_values.brand || ""}`, key_values);
        const combinedTitleAndKey = `${title} ${key_values.model || ""}`.trim();
        const modelCandidates = extractModelCandidates(combinedTitleAndKey, brand);
        
        let model = null;
        let modelError = null;

        if (modelCandidates.length === 1) {
            model = modelCandidates[0];
        } else if (modelCandidates.length > 1) {
            modelError = `Multiple conflicting GPU models found in listing: [${modelCandidates.join(", ")}]. Please select the exact model manually.`;
        } else {
            const gpuLikeMatch = combinedTitleAndKey.match(/\b(?:RTX|GTX|RX|GTS|GT|ARC)\s*\d{3,4}(?:\s*(?:TI|XT|XTX|SUPER))?\b/i);
            if (gpuLikeMatch) {
                modelError = `Unrecognized GPU model: "${gpuLikeMatch[0]}". Please select a supported model from the dataset.`;
            } else {
                modelError = "Could not identify a recognized GPU model from the listing. Please select the model manually.";
            }
        }

        const vramResult = extractVram(pageContext, model);
        const manufacturer = extractManufacturer(model, brand);

        const missingFields = [];
        let specError = null;

        if (!model) {
            missingFields.push("GPU Model");
        }

        if (!vramResult.vram_gb) {
            missingFields.push("VRAM (GB)");
        } else if (!vramResult.is_valid && model && vramResult.spec) {
            specError = `Listed VRAM (${vramResult.vram_gb} GB) does not match official specifications for ${model} (valid options: ${vramResult.spec.valid_vrams.join("/ ")} GB). Please verify specifications.`;
            missingFields.push(`Valid VRAM for ${model}`);
        }

        if (!price || isNaN(price) || price <= 0) {
            missingFields.push("Listing Price");
        }

        const isValid = missingFields.length === 0;
        const errorMessage = specError || modelError || (isValid ? null : `Missing [${missingFields.join(", ")}]. Please select details manually.`);

        return {
            category: "gpu",
            valid: isValid,
            missing_fields: missingFields,
            error_message: errorMessage,
            manual_selection_required: !isValid,
            detected_models: modelCandidates,
            vram_auto_filled: vramResult.is_auto_filled,
            data: {
                title: title,
                model: model || "",
                vram_gb: vramResult.vram_gb || null,
                brand: brand || "Any",
                manufacturer: manufacturer || "Any",
                listed_price: price || null,
                stock: "In Stock",
                description: raw_text || title || ""
            }
        };
    }

    return {
        parse: parse,
        extractModel: extractModel,
        extractModelCandidates: extractModelCandidates,
        extractVram: extractVram,
        extractBrand: extractBrand,
        CANONICAL_MODELS: CANONICAL_MODELS,
        STANDALONE_MODEL_RULES: STANDALONE_MODEL_RULES,
        GPU_SPECS_DATASET: GPU_SPECS_DATASET,
        KNOWN_BRANDS: KNOWN_BRANDS
    };
})();
