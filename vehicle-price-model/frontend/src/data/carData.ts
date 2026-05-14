export interface VariantData {
  years: number[];
  transmissions: string[];
  fuelTypes: string[];
  basePrice: number;
  yearFactor: number; // Price change per year
}

export const VEHICLE_MODELS = ["Toyota Corolla", "Toyota Aqua", "Suzuki Alto"];

export const VARIANTS: Record<string, VariantData> = {
  "121": {
    years: Array.from({ length: 9 }, (_, i) => 2000 + i), // 2000-2008
    transmissions: ["Manual", "Automatic"],
    fuelTypes: ["Petrol", "Diesel"],
    basePrice: 5500000,
    yearFactor: 100000
  },
  "141": {
    years: Array.from({ length: 7 }, (_, i) => 2007 + i), // 2007-2013
    transmissions: ["Manual", "Automatic"],
    fuelTypes: ["Petrol"],
    basePrice: 6500000,
    yearFactor: 150000
  },
  "AE110": {
    years: Array.from({ length: 6 }, (_, i) => 1995 + i), // 1995-2000
    transmissions: ["Manual", "Automatic"],
    fuelTypes: ["Petrol", "Diesel"],
    basePrice: 3500000,
    yearFactor: 80000
  },
  "DX/KE72": {
    years: Array.from({ length: 11 }, (_, i) => 1980 + i), // 1980-1990
    transmissions: ["Manual"],
    fuelTypes: ["Petrol"],
    basePrice: 1200000,
    yearFactor: 50000
  },
  "Aqua": {
    years: [2012, 2013, 2014, 2015],
    transmissions: ["Automatic"],
    fuelTypes: ["Hybrid"],
    basePrice: 6500000,
    yearFactor: 300000
  },
  // Suzuki Alto Groups
  "G1_Manual_2000-2012": {
    years: [2000, 2001, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
    transmissions: ["Manual"],
    fuelTypes: ["Petrol"],
    basePrice: 2800000,
    yearFactor: 80000
  },
  "G2_Manual_2013-2015": {
    years: [2013, 2014, 2015],
    transmissions: ["Manual"],
    fuelTypes: ["Petrol"],
    basePrice: 3500000,
    yearFactor: 150000
  },
  "G3_Manual_2016-2019": {
    years: [2016, 2017, 2018, 2019],
    transmissions: ["Manual"],
    fuelTypes: ["Petrol"],
    basePrice: 4000000,
    yearFactor: 200000
  },
  "G4_Auto_lt700_2000-2015": {
    years: [2002, 2003, 2004, 2005, 2008, 2009],
    transmissions: ["Automatic"],
    fuelTypes: ["Petrol"],
    basePrice: 3200000,
    yearFactor: 150000
  }
};


export interface ModelMetrics {
  name: string;
  mae: number;
  r2: number;
}

export const ML_MODELS: ModelMetrics[] = [
  {
    name: "Random Forest Regressor",
    mae: 405792,
    r2: 0.861
  },
  {
    name: "XGBoost Regressor",
    mae: 417980,
    r2: 0.851
  },
  {
    name: "Gradient Boosting",
    mae: 335585,
    r2: 0.326
  }
];
