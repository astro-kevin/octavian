MINIMUM_STARS_PER_GALAXY = 16
MINIMUM_DM_PER_HALO = 24
MINIMUM_GAS_PER_CLOUD = 16

nHlim = 0.13
Tlim = 1.e5
XH = 0.76

ptypes = {
  'gas': 'PartType0',
  'dm': 'PartType1',
  'star': 'PartType4',
  'bh': 'PartType5',
}

ptype_ids = {
  'gas': 0,
  'dm': 1,
  'star': 4,
  'bh': 5,
}

ptype_names = {
  0: 'gas',
  1: 'dm',
  4: 'star',
  5: 'bh'
}

ptype_lists = {
  'gas': 'glist',
  'dm': 'dmlist',
  'star': 'slist',
  'bh': 'bhlist',
}

prop_aliases = {
  'pos':'Coordinates',
  'vel':'Velocities',
  'mass':'Masses',
  'u':'InternalEnergy',
  'temp':'InternalEnergy',
  'temperature':'InternalEnergy',
  'rho':'Density',
  'nh':'NeutralHydrogenAbundance',
  'sfr':'StarFormationRate',
  'metallicity':'Metallicity',
  'age':'StellarFormationTime',
  'pot':'Potential',
  'fh2':'FractionH2',
  'bhmass': 'BH_Mass',
  'bhmdot': 'BH_Mdot',
  'pid': 'ParticleIDs',
}

prop_units = {
  'pos':'kpc*a/h',
  'vel':'km/s*a**(1/2)',
  'mass':'1.e10 * Msun/h',
  'u':'K',
  'temp':'K',
  'temperature':'K',
  'rho':'1.e10 * Msun/h / (kpc*a/h)**3',
  'nh':'1.e10 * Msun/h',
  'sfr':'1.e10 * Msun/h/s',
  'bhmass': '1.e10 * Msun/h',
  'bhmdot': '1.e10 * Msun/h/s',
}

code_units = {
  'pos':'kpc*a',
  'vel':'km/s',
  'mass':'Msun',
  'u':'K',
  'temp':'K',
  'temperature':'K',
  'rho':'g/cm**3',
  'rhocrit': 'Msun/kpc**3',
  'nh':'Msun',
  'sfr':'Msun/s',
  'bhmass': 'Msun',
  'bhmdot': 'Msun/s',
}

prop_columns = {
  'pos':['x', 'y', 'z'],
  'vel':['vx', 'vy', 'vz'],
  'pot':'potential',
  'rho':'rho',
  'hsml':'hsml',
  'sfr':'sfr',
  'mass':'mass',
  'u':'temperature',
  'temp':'temperature',
  'temperature':'temperature',
  'ne':'ne',
  'nh':'nh',
  'fHI':'fHI',
  'pid':'pid',
  'fh2':'fH2',
  'metallicity':'metallicity',
  'age':'age',
  'bhmdot':'bhmdot',
  'bhmass': 'mass',
}
