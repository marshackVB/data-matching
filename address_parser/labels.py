# Labeled address data for training

label_types = ['AddressNumber', 'StreetNamePreType', 'StreetName', 'StreetNamePostType',
               'StreetNamePreDirectional', 'OccupancyType', 'OccupancyIdentifier', 'USPSBoxType',
               'USPSBoxID', 'StreetNamePostDirectional', 'SubaddressType', 'SubaddressIdentifier']

usaddress_labels = [
                   'AddressNumberPrefix',
                   'AddressNumber',
                   'AddressNumberSuffix',
                   'StreetNamePreModifier',
                   'StreetNamePreDirectional',
                   'StreetNamePreType',
                   'StreetName',
                   'StreetNamePostType',
                   'StreetNamePostDirectional',
                   'SubaddressType',
                   'SubaddressIdentifier',
                   'BuildingName',
                   'OccupancyType',
                   'OccupancyIdentifier',
                   'CornerOf',
                   'LandmarkName',
                   'PlaceName',
                   'StateName',
                   'ZipCode',
                   'USPSBoxType',
                   'USPSBoxID',
                   'USPSBoxGroupType',
                   'USPSBoxGroupID',
                   'IntersectionSeparator',
                   'Recipient',
                   'NotAddress',
               ]

DIRECTIONS = set(['n', 's', 'e', 'w',
                  'ne', 'nw', 'se', 'sw',
                  'north', 'south', 'east', 'west',
                  'northeast', 'northwest', 'southeast', 'southwest'])

STREET_NAMES = {
    'allee', 'alley', 'ally', 'aly', 'anex', 'annex', 'annx', 'anx',
    'arc', 'arcade', 'av', 'ave', 'aven', 'avenu', 'avenue', 'avn', 'avnue',
    'bayoo', 'bayou', 'bch', 'beach', 'bend', 'bg', 'bgs', 'bl', 'blf',
    'blfs', 'bluf', 'bluff', 'bluffs', 'blvd', 'bnd', 'bot', 'bottm',
    'bottom', 'boul', 'boulevard', 'boulv', 'br', 'branch', 'brdge', 'brg',
    'bridge', 'brk', 'brks', 'brnch', 'brook', 'brooks', 'btm', 'burg',
    'burgs', 'byp', 'bypa', 'bypas', 'bypass', 'byps', 'byu', 'camp', 'canyn',
    'canyon', 'cape', 'causeway', 'causwa', 'causway', 'cen', 'cent',
    'center', 'centers', 'centr', 'centre', 'ci', 'cir', 'circ', 'circl',
    'circle', 'circles', 'cirs', 'ck', 'clb', 'clf', 'clfs', 'cliff',
    'cliffs', 'club', 'cmn', 'cmns', 'cmp', 'cnter', 'cntr', 'cnyn', 'common',
    'commons', 'cor', 'corner', 'corners', 'cors', 'course', 'court',
    'courts', 'cove', 'coves', 'cp', 'cpe', 'cr', 'crcl', 'crcle', 'crecent',
    'creek', 'cres', 'crescent', 'cresent', 'crest', 'crk', 'crossing',
    'crossroad', 'crossroads', 'crscnt', 'crse', 'crsent', 'crsnt', 'crssing',
    'crssng', 'crst', 'crt', 'cswy', 'ct', 'ctr', 'ctrs', 'cts', 'curv',
    'curve', 'cv', 'cvs', 'cyn', 'dale', 'dam', 'div', 'divide', 'dl', 'dm',
    'dr', 'driv', 'drive', 'drives', 'drs', 'drv', 'dv', 'dvd', 'est',
    'estate', 'estates', 'ests', 'ex', 'exp', 'expr', 'express', 'expressway',
    'expw', 'expy', 'ext', 'extension', 'extensions', 'extn', 'extnsn',
    'exts', 'fall', 'falls', 'ferry', 'field', 'fields', 'flat', 'flats',
    'fld', 'flds', 'fls', 'flt', 'flts', 'ford', 'fords', 'forest', 'forests',
    'forg', 'forge', 'forges', 'fork', 'forks', 'fort', 'frd', 'frds',
    'freeway', 'freewy', 'frg', 'frgs', 'frk', 'frks', 'frry', 'frst', 'frt',
    'frway', 'frwy', 'fry', 'ft', 'fwy', 'garden', 'gardens', 'gardn',
    'gateway', 'gatewy', 'gatway', 'gdn', 'gdns', 'glen', 'glens', 'gln',
    'glns', 'grden', 'grdn', 'grdns', 'green', 'greens', 'grn', 'grns',
    'grov', 'grove', 'groves', 'grv', 'grvs', 'gtway', 'gtwy', 'harb',
    'harbor', 'harbors', 'harbr', 'haven', 'havn', 'hbr', 'hbrs', 'height',
    'heights', 'hgts', 'highway', 'highwy', 'hill', 'hills', 'hiway', 'hiwy',
    'hl', 'hllw', 'hls', 'hollow', 'hollows', 'holw', 'holws', 'hrbor', 'ht',
    'hts', 'hvn', 'hway', 'hwy', 'inlet', 'inlt', 'is', 'island', 'islands',
    'isle', 'isles', 'islnd', 'islnds', 'iss', 'jct', 'jction', 'jctn',
    'jctns', 'jcts', 'junction', 'junctions', 'junctn', 'juncton', 'key',
    'keys', 'knl', 'knls', 'knol', 'knoll', 'knolls', 'ky', 'kys', 'la',
    'lake', 'lakes', 'land', 'landing', 'lane', 'lanes', 'lck', 'lcks', 'ldg',
    'ldge', 'lf', 'lgt', 'lgts', 'light', 'lights', 'lk', 'lks', 'ln', 'lndg',
    'lndng', 'loaf', 'lock', 'locks', 'lodg', 'lodge', 'loop', 'loops', 'lp',
    'mall', 'manor', 'manors', 'mdw', 'mdws', 'meadow', 'meadows', 'medows',
    'mews', 'mi', 'mile', 'mill', 'mills', 'mission', 'missn', 'ml', 'mls',
    'mn', 'mnr', 'mnrs', 'mnt', 'mntain', 'mntn', 'mntns', 'motorway',
    'mount', 'mountain', 'mountains', 'mountin', 'msn', 'mssn', 'mt', 'mtin',
    'mtn', 'mtns', 'mtwy', 'nck', 'neck', 'opas', 'orch', 'orchard', 'orchrd',
    'oval', 'overlook', 'overpass', 'ovl', 'ovlk', 'park', 'parks', 'parkway',
    'parkways', 'parkwy', 'pass', 'passage', 'path', 'paths', 'pike', 'pikes',
    'pine', 'pines', 'pk', 'pkway', 'pkwy', 'pkwys', 'pky', 'pl', 'place',
    'plain', 'plaines', 'plains', 'plaza', 'pln', 'plns', 'plz', 'plza',
    'pne', 'pnes', 'point', 'points', 'port', 'ports', 'pr', 'prairie',
    'prarie', 'prk', 'prr', 'prt', 'prts', 'psge', 'pt', 'pts', 'pw', 'pwy',
    'rad', 'radial', 'radiel', 'radl', 'ramp', 'ranch', 'ranches', 'rapid',
    'rapids', 'rd', 'rdg', 'rdge', 'rdgs', 'rds', 'rest', 'ri', 'ridge',
    'ridges', 'rise', 'riv', 'river', 'rivr', 'rn', 'rnch', 'rnchs', 'road',
    'roads', 'route', 'row', 'rpd', 'rpds', 'rst', 'rte', 'rue', 'run', 'rvr',
    'shl', 'shls', 'shoal', 'shoals', 'shoar', 'shoars', 'shore', 'shores',
    'shr', 'shrs', 'skwy', 'skyway', 'smt', 'spg', 'spgs', 'spng', 'spngs',
    'spring', 'springs', 'sprng', 'sprngs', 'spur', 'spurs', 'sq', 'sqr',
    'sqre', 'sqrs', 'sqs', 'squ', 'square', 'squares', 'st', 'sta', 'station',
    'statn', 'stn', 'str', 'stra', 'strav', 'strave', 'straven', 'stravenue',
    'stravn', 'stream', 'street', 'streets', 'streme', 'strm', 'strt',
    'strvn', 'strvnue', 'sts', 'sumit', 'sumitt', 'summit', 'te', 'ter',
    'terr', 'terrace', 'throughway', 'tl', 'tpk', 'tpke', 'tr', 'trace',
    'traces', 'track', 'tracks', 'trafficway', 'trail', 'trailer', 'trails',
    'trak', 'trce', 'trfy', 'trk', 'trks', 'trl', 'trlr', 'trlrs', 'trls',
    'trnpk', 'trpk', 'trwy', 'tunel', 'tunl', 'tunls', 'tunnel', 'tunnels',
    'tunnl', 'turn', 'turnpike', 'turnpk', 'un', 'underpass', 'union',
    'unions', 'uns', 'upas', 'valley', 'valleys', 'vally', 'vdct', 'via',
    'viadct', 'viaduct', 'view', 'views', 'vill', 'villag', 'village',
    'villages', 'ville', 'villg', 'villiage', 'vis', 'vist', 'vista', 'vl',
    'vlg', 'vlgs', 'vlly', 'vly', 'vlys', 'vst', 'vsta', 'vw', 'vws', 'walk',
    'walks', 'wall', 'way', 'ways', 'well', 'wells', 'wl', 'wls', 'wy', 'xc',
    'xg', 'xing', 'xrd', 'xrds'
    }

labels = [
          [('123', 'AddressNumber'), ('Main', 'StreetName'), ('Street', 'StreetNamePostType')],
          [('60', 'AddressNumber'), ('Railroad', 'StreetName'), ('Avenue', 'StreetNamePostType')],
          [('3520', 'AddressNumber'), ('E.', 'StreetNamePreDirectional'), ('Ave.', 'StreetNamePreType'), ('M', 'StreetName')],
          [('3255', 'AddressNumber'), ('Highway','StreetNamePreType'), ('345','StreetName')],
          [('32', 'AddressNumber'), ('Highway','StreetNamePreType'), ('345','StreetName')],
          [('1', 'AddressNumber'), ('Highway','StreetNamePreType'), ('1','StreetName')],
          [('967', 'AddressNumber'), ('Highway','StreetNamePreType'), ('301','StreetName')],
          [('118', 'AddressNumber'), ('Thompson', 'StreetName'), ('Dr.', 'StreetNamePostType')],
          [('6150', 'AddressNumber'), ('Brookshire', 'StreetName'), ('Blvd','StreetNamePostType'), ('#H', 'OccupancyIdentifier')],
          [('600', 'AddressNumber'), ('Crawford', 'StreetName'), ('Street','StreetNamePostType'), ('Suite', 'OccupancyType'), ('300', 'OccupancyIdentifier')],
          [('25', 'AddressNumber'), ('Energy', 'StreetName'), ('Drive','StreetNamePostType')],
          [('2313', 'AddressNumber'), ('E.', 'StreetNamePreDirectional'), ('92nd', 'StreetName'), ('St.','StreetNamePostType')],
          [('8957', 'AddressNumber'), ('Kingsridge', 'StreetName'), ('Drive','StreetNamePostType')],
          [('233', 'AddressNumber'), ('Spur', 'StreetName'), ('Road', 'StreetNamePostType')],
          [('1318', 'AddressNumber'), ('Hatton', 'StreetName'), ('Road', 'StreetNamePostType')],
          [('1543', 'AddressNumber'), ('Mission', 'StreetName'), ('Street','StreetNamePostType')],
          [('2211', 'AddressNumber'), ('East', 'StreetNamePreDirectional'), ('Division', 'StreetName')],
          [('68', 'AddressNumber'), ('Blueberry', 'StreetName'), ('Lane','StreetNamePostType'), ('#13', 'OccupancyIdentifier')],
          [('859', 'AddressNumber'), ('W.', 'StreetNamePreDirectional'), ('South', 'StreetName'), ('Jordan', 'StreetName'), ('Parkway', 'StreetNamePostType'), ('#72', 'OccupancyIdentifier')],
          [('8730', 'AddressNumber'), ('NC', 'StreetName'), ('183', 'StreetName')],
          [('11', 'AddressNumber'), ('Eastgate', 'StreetName'), ('Avenue', 'StreetNamePostType')],
          [('6860', 'AddressNumber'), ('W', 'StreetNamePreDirectional'), ('115th', 'StreetName'), ('St', 'StreetNamePostType')],
          [('7640', 'AddressNumber'), ('Wallace', 'StreetName'), ('Road','StreetNamePostType')],
          [('2535', 'AddressNumber'), ('Walnut', 'StreetName'), ('Hill', 'StreetName'), ('Lane', 'StreetNamePostType')],
          [('4293', 'AddressNumber'), ('South', 'StreetNamePreDirectional'), ('M-37', 'StreetName')],
          [('5407', 'AddressNumber'), ('Summerville', 'StreetName'), ('Road', 'StreetNamePostType')],
          [('251', 'AddressNumber'), ('AA', 'StreetName'), ('Highway', 'StreetNamePostType')],
          [('7459', 'AddressNumber'), ('W.', 'StreetNamePreDirectional'), ('Central', 'StreetName'), ('Avenue', 'StreetNamePostType')],
          [('3420', 'AddressNumber'), ('Clark', 'StreetName'), ('Lane', 'StreetNamePostType')],
          [('4', 'AddressNumber'), ('Sibley', 'StreetName'), ('Blvd', 'StreetNamePostType')],
          [('PO', 'USPSBoxType',), ('Box', 'USPSBoxType',), ('#234', 'USPSBoxID')],
          [('PO', 'USPSBoxType',), ('Box', 'USPSBoxType',), ('5298', 'USPSBoxID')],
          [('PO', 'USPSBoxType',), ('Box', 'USPSBoxType',), ('789', 'USPSBoxID')],
          [('PO', 'USPSBoxType',), ('Box', 'USPSBoxType',), ('23', 'USPSBoxID')],
          [('5101', 'AddressNumber'), ('Hinkleville', 'StreetName'), ('Road', 'StreetNamePostType'),  ('Space', 'OccupancyType'), ('#540', 'OccupancyIdentifier')],
          [('14122', 'AddressNumber'), ('W.', 'StreetNamePreDirectional'), ('McDowell', 'StreetName'), ('Rd','StreetNamePostType'), ('Ste','OccupancyType'), ('100', 'OccupancyIdentifier')],
          [('28972','AddressNumber'), ('W.','StreetNamePreDirectional'), ('IL','StreetNamePreType'), ('Route','StreetNamePreType'), ('120','StreetName')],
          [('28972','AddressNumber'), ('N.','StreetNamePreDirectional'), ('IN','StreetNamePreType'), ('Highway','StreetNamePreType'), ('1','StreetName')],
          [('28972','AddressNumber'), ('SW.','StreetNamePreDirectional'), ('AZ','StreetNamePreType'), ('Route','StreetNamePreType'), ('20','StreetName')],
          [('8550','AddressNumber'), ('NW','StreetNamePreDirectional'), ('44','StreetName'), ('ST','StreetNamePostType')],
          [('4211','AddressNumber'), ('W','StreetNamePreDirectional'), ('59','StreetName'), ('ST','StreetNamePostType')],
          [('103','AddressNumber'), ('S','StreetNamePreDirectional'), ('US','StreetNamePreType'), ('1','StreetName'), ('#F2','OccupancyIdentifier')],
          [('5068','AddressNumber'), ('ANNUNCIATION','StreetName'), ('CIR','StreetNamePostType'), ('#106','OccupancyIdentifier')],
          [('652','AddressNumber'), ('BEAL','StreetName'), ('PKWY','StreetNamePostType'), ('#F','OccupancyIdentifier')],
          [('4751','AddressNumber'), ('OLD','StreetName'), ('GOLDENROD','StreetName'), ('RD','StreetNamePostType'), ('#1','OccupancyIdentifier')],
          [('1462','AddressNumber'), ('N','StreetNamePreDirectional'), ('ROCK','StreetName'), ('SPRING','StreetName'), ('ROAD','StreetNamePostType')],
          [('235','AddressNumber'), ('APOLLO','StreetName'), ('BEACH','StreetName'), ('BLVD','StreetNamePostType'), ('#163','OccupancyIdentifier')],
          [('1620','AddressNumber'), ('W','StreetNamePreDirectional'), ('UNIVERSITY','StreetName'), ('AVENUE','StreetNamePostType'), ('SUITE','OccupancyType'), ('A','OccupancyIdentifier')],
          [('13770','AddressNumber'), ('W','StreetNamePreDirectional'), ('COLONIAL','StreetName'), ('DR','StreetNamePostType'), ('#','OccupancyIdentifier'), ('160','OccupancyIdentifier')],
          [('13770','AddressNumber'),('WILLIAMS','StreetName'), ('AVE','StreetNamePostType'), ('#','OccupancyIdentifier'), ('12','OccupancyIdentifier')],
          [('13770','AddressNumber'), ('SW','StreetNamePreDirectional'), ('FEDERAL','StreetName'), ('HWY','StreetNamePostType'), ('#','OccupancyIdentifier'), ('2','OccupancyIdentifier')],
          [('6303','AddressNumber'), ('-','AddressNumber'), ('6305A','AddressNumber'), ('MIRAMAR','StreetName'), ('PKWY','StreetNamePostType')],
          [('3110','AddressNumber'), ('SR','StreetNamePreType'), ('674','StreetName')],
          [('876','AddressNumber'), ('SR','StreetNamePreType'), ('94','StreetName')],
          [('9843','AddressNumber'), ('SR','StreetNamePreType'), ('104','StreetName')],
          [('2643','AddressNumber'), ('GULF','StreetName'), ('TO','StreetName'), ('BAY','StreetName'), ('BLVD','StreetNamePostType'), ('STE','OccupancyType'),('1550-1560','OccupancyIdentifier')],
          [('10233','AddressNumber'), ('OKEECHOBEE','StreetName'), ('BLVD','StreetNamePostType'), ('B-11','OccupancyIdentifier')],
          [('3601','AddressNumber'), ('W','StreetNamePreDirectional'), ('COMMERCIAL','StreetName'), ('BLVD','StreetNamePostType'), ('UNIT','OccupancyType'), ('2','OccupancyIdentifier')],
          [('9802','AddressNumber'), ('BAYMEADOWS','StreetName'), ('ROAD','StreetNamePostType'), ('SUITE','OccupancyType'), ('3','OccupancyIdentifier')],
          [('1355','AddressNumber'), ('MARKET','StreetName'), ('ST','StreetNamePostType'), ('A7','OccupancyIdentifier')],
          [('642','AddressNumber'), ('N','StreetNamePreDirectional'), ('DIXIE','StreetName'), ('FWY','StreetNamePostType')],
          [('99-10','AddressNumber'), ('43rd','StreetName'), ('Avenue','StreetNamePostType')],
          [('3000','AddressNumber'), ('Highway','StreetNamePreType'), ('41','StreetName')],
          [('5651','AddressNumber'), ('CR','StreetNamePreType'), ('4100','StreetName')],
          [('804','AddressNumber'), ('US','StreetNamePreType'), ('1','StreetName'), ('#10','OccupancyIdentifier')],
          [('515','AddressNumber'), ('SW','StreetNamePreDirectional'), ('PARK','StreetName'), ('ST','StreetNamePostType')],
          [('13490','AddressNumber'), ('ORANGE','StreetName'), ('AVE','StreetNamePostType')],
          [('550','AddressNumber'), ('W','StreetNamePreDirectional'), ('HILLSBOROUGH','StreetName'), ('AVE','StreetNamePostType')],
          [('255','AddressNumber'), ('MIRACLE','StreetName'), ('STRIP','StreetName'), ('PKWY','StreetNamePostType'), ('SE','StreetNamePostDirectional')],
          [('901','AddressNumber'), ('TIMBERLINE','StreetName'), ('DRIVE','StreetNamePostType')],
          [('1905','AddressNumber'), ('S','StreetNamePreDirectional'), ('FRENCH','StreetName'), ('AVE','StreetNamePostType')],
          [('12655','AddressNumber'), ('S','StreetNamePreDirectional'), ('DIXIE','StreetName'), ('HWY','StreetNamePostType')],
          [('7335','AddressNumber'), ('RADIO','StreetName'), ('RD','StreetNamePostType'), ('SUITE','OccupancyType'), ('109', 'OccupancyIdentifier')],
          [('3601','AddressNumber'), ('W','StreetNamePreDirectional'), ('COMMERCIAL','StreetName'), ('BLVD','StreetNamePostType'), ('UNIT','OccupancyType'), ('2','OccupancyIdentifier')],
          [('1717','AddressNumber'), ('N','StreetNamePreDirectional'), ('BAYSHORE','StreetName'), ('DR','StreetNamePostType'), ('#','OccupancyIdentifier'), ('101','OccupancyIdentifier')],
          [('10816','AddressNumber'), ('US','StreetNamePreType'), ('41','StreetName'), ('N','StreetNamePostDirectional')],
          [('123','AddressNumber'), ('US','StreetNamePreType'), ('1','StreetName'), ('SW','StreetNamePostDirectional')],
          [('22','AddressNumber'), ('US','StreetNamePreType'), ('234','StreetName'), ('S','StreetNamePostDirectional')],
          [('619','AddressNumber'), ('N.','StreetNamePreDirectional'), ('PINELLAS','StreetName'), ('AVE','StreetNamePostType')],
          [('9802','AddressNumber'), ('BAYMEADOWS','StreetName'), ('ROAD','StreetNamePostType'), ('SUITE','StreetNamePostType'), ('3','OccupancyIdentifier')],
          [('1005','AddressNumber'), ('SPRING','StreetName'), ('VILLAS','StreetName'), ('POINT','StreetNamePostType')],
          [('6954','AddressNumber'), ('COLLINS','StreetName'), ('AVE','StreetNamePostType')],
          [('2410','AddressNumber'), ('N','StreetNamePreDirectional'), ('FEDERAL','StreetName'), ('HWY','StreetNamePostType')],
          [('3215','AddressNumber'), ('S','StreetNamePreDirectional'), ('US','StreetNamePreType'), ('1','StreetName'), ('STE', 'OccupancyType'), ('H','OccupancyIdentifier')],
          [('1532','AddressNumber'), ('S','StreetNamePreDirectional'), ('PINE','StreetName'), ('AVE','StreetNamePostType')],
          [('1355','AddressNumber'), ('MARKET','StreetName'), ('ST','StreetNamePostType'), ('A7','OccupancyIdentifier')],
          [('642','AddressNumber'), ('N','StreetNamePreDirectional'), ('DIXIE','StreetName'), ('FWY','StreetNamePostType')],
          [('21','AddressNumber'), ('JEFFERSON', 'StreetName'), ('AVENUE', 'StreetNamePostType'), ('N', 'StreetNamePostDirectional'), ('APT', 'OccupancyType'), ('#', 'OccupancyIdentifier'), ('12', 'OccupancyIdentifier')],
          [('243','AddressNumber'), ('KIAPAI', 'StreetName'), ('CIR', 'StreetNamePostType'), ('SE', 'StreetNamePostDirectional'), ('#23', 'OccupancyIdentifier')],
          [('8733','AddressNumber'), ('PARK', 'StreetName'), ('ST', 'StreetNamePostType'), ('NW', 'StreetNamePostDirectional'), ('APT','OccupancyType'), ('#2', 'OccupancyIdentifier')],
          [('UNE', 'OccupancyType')],
          [('UNK', 'OccupancyType')],
          [('STE400', 'OccupancyIdentifier')],
          [('1410', 'AddressNumber'), ('AVE', 'StreetNamePreType'), ('J', 'StreetName')],
          [('7538', 'AddressNumber'), ('AVE', 'StreetNamePreType'), ('T', 'StreetName')],
          [('5913', 'AddressNumber'), ('AVE', 'StreetNamePreType'), ('B', 'StreetName')],
          [('947', 'AddressNumber'), ('HWY', 'StreetNamePreType'), ('34E', 'StreetName')],
          [('180', 'AddressNumber'), ('HWY', 'StreetNamePreType'), ('A1A', 'StreetName')],
          [('254', 'AddressNumber'), ('HWY', 'StreetNamePreType'), ('A1A', 'StreetName')],
          [('1013', 'AddressNumber'), ('HWY', 'StreetNamePreType'), ('59W', 'StreetName')],
          [('1515', 'AddressNumber'), ('EAST', 'StreetNamePreDirectional'), ('9TH', 'StreetName')],
          [('201', 'AddressNumber'), ('NORTH', 'StreetNamePreDirectional'), ('14TH', 'StreetName')],
          [('1715', 'AddressNumber'), ('CAPE', 'StreetName'), ('CORAL', 'StreetName'), ('PARKWAY', 'StreetNamePostType'), ('WEST', 'StreetNamePostDirectional')],
          [('2468', 'AddressNumber'), ('US', 'StreetName'), ('HWY', 'StreetNamePostDirectional'), ('441/27', 'StreetName'), ('101', 'OccupancyIdentifier')],
          [('8685', 'AddressNumber'), ('CR', 'StreetNamePreType'), ('466A', 'StreetName')],
          [('2105', 'AddressNumber'), ('CR', 'StreetNamePreType'), ('540A', 'StreetName')],
          [('U', 'StreetName'), ('S', 'StreetName'), ('31', 'StreetName')],
          [('U', 'StreetName'), ('S', 'StreetName'), ('36', 'StreetName')],
          [('N', 'StreetNamePreDirectional'), ('US1', 'StreetName')],
          [('2065', 'AddressNumber'), ('SOUTH', 'StreetNamePreDirectional'), ('US', 'StreetName'), ('HWY', 'StreetNamePreType'), ('1', 'StreetName')],
          [('3806', 'AddressNumber'), ('AVE', 'StreetNamePreType'), ('I', 'StreetName'), ('STE', 'OccupancyType'), ('11', 'OccupancyIdentifier')],
          [('1781', 'AddressNumber'), ('DUNLAWTON', 'StreetName'), ('AVE', 'StreetNamePostType'), ('OUTPARCEL', 'OccupancyType'), ('4', 'OccupancyIdentifier')],
          [('PMB', 'SubaddressType'), ('519', 'SubaddressIdentifier'), ('BOX', 'USPSBoxType'), ('10000', 'USPSBoxID')],
          [('PMB', 'SubaddressType'), ('246', 'SubaddressIdentifier'), ('BOX', 'USPSBoxType'), ('100', 'USPSBoxID')],
          [('PMB', 'SubaddressType'), ('665', 'SubaddressIdentifier'), ('BOX', 'USPSBoxType'), ('10001', 'USPSBoxID')],
          [('CALL', 'USPSBoxType'), ('BOX', 'USPSBoxType'), ('51990', 'USPSBoxID')],
          [('215', 'AddressNumber'), ('SOUTH', 'StreetNamePreDirectional'), ('A1A', 'StreetName')],
          [('1106', 'AddressNumber'), ('ROUTE', 'StreetNamePreType'), ('9W', 'StreetName')],
          [('P', 'USPSBoxType'), ('O', 'USPSBoxType'), ('BOX', 'USPSBoxType'), ('75', 'USPSBoxID')],
          [('46218', 'AddressNumber'), ('I', 'StreetNamePreType'), ('H', 'StreetNamePreType'), ('10', 'StreetName')],
          [('2930', 'AddressNumber'),  ('WATERFRONT', 'StreetName'),      ('PARKWAY', 'StreetNamePostType')],
          [('843', 'AddressNumber'),   ('SEBASTIAN', 'StreetName'),       ('STREET', 'StreetNamePostType')],
          [('179', 'AddressNumber'),   ('MEDFORD', 'StreetName'),         ('AVE', 'StreetNamePostType')],
          [('1004', 'AddressNumber'),  ('HIGHWAY', 'StreetNamePreType'),  ('A1A', 'StreetName')],
          [('20660', 'AddressNumber'), ('HWY', 'StreetNamePreType'),      ('63', 'StreetName')],
          [('903', 'AddressNumber'),   ('E', 'StreetNamePreDirectional'), ('US', 'StreetNamePreType'), ('HIGHWAY', 'StreetNamePreType'), ('80', 'StreetName')],
          [('54', 'AddressNumber'),    ('N', 'StreetNamePreDirectional'), ('GROESBECK', 'StreetName')],
          [('P', 'USPSBoxType'),       ('O', 'USPSBoxType'), ('BOX', 'USPSBoxType'), ('500029', 'USPSBoxID')],
          [('2675', 'AddressNumber'),  ('100TH', 'StreetName'), ('STREET', 'StreetNamePostType')],
          [('54', 'AddressNumber'),    ('N', 'StreetNamePreDirectional'), ('GROESBECK', 'StreetName')],
          [('2402', 'AddressNumber'),  ('BROCK', 'StreetName'), ('ST', 'StreetNamePostType'), ('A', 'OccupancyIdentifier')],
          [('2120', 'AddressNumber'),  ('EAST', 'StreetNamePreDirectional'), ('3900', 'StreetName'), ('SOUTH', 'StreetNamePostDirectional')],
          [('635', 'AddressNumber'),   ('S', 'StreetNamePreType'), ('TROOPER', 'StreetName'), ('RD', 'StreetNamePostType')]
         ]


def format_address_list(x):
    ''"Format a list of address for labeling"""
    splitter = lambda x: x.split()
    t_split = [splitter(i) for i in x]
    return [[('{0}'.format(x), ) for x in l] for l in t_split]
