#pragma once

/*  T [K]                    aerosol conc. fact.       air density [kg/m^3]
    301.1500000000000         1.000000000000000         1.167964615529032     
    300.5291262558619         1.000000000000000         1.150449720531184     
    299.5484160885035         1.000000000000000         1.141087409674031     
    298.5676927597234         1.000000000000000         1.131770836523778     
    297.5869561823087         1.000000000000000         1.122499926975512     
    297.2070025020243         1.000000000000000         1.111024151109086     
    296.6974450335807         1.000000000000000         1.100131508534285     
    296.1961071611942         1.000000000000000         1.089296873796241     
    295.6931749189547         1.000000000000000         1.078556570625417     
    295.1983702402352         1.000000000000000         1.067874644671132     
    294.7600668897709         1.000000000000000         1.057078088564207     
    294.3199523985062         1.000000000000000         1.046381212479095     
    293.8684073023192         1.000000000000000         1.035817012238898     
    293.4150894977445         1.000000000000000         1.025349568943642     
    292.8357409531135         1.000000000000000         1.015408695017796     
    292.2359865541015         1.000000000000000         1.005614897689672     
    291.6634284223253         1.000000000000000        0.9958027537816577     
    291.0894892174012         1.000000000000000        0.9860719172834188     
    290.5047400479028        0.9610532350560985        0.9764534662322681     
    289.9186451731319        0.9152870540745627        0.9669141140649613     
    289.3312063492775        0.8715965240057739        0.9574532680076016     
    288.7330899755047        0.8290917809871519        0.9481009930467961     
    288.1336646839490        0.7885691895200798        0.9388251106651607     
    287.5236601633937        0.7492219896966925        0.9296550366246522     
    287.1156688387720        0.7268910657539789        0.9199075280740460     
    286.7057142896359        0.7050930914993466        0.9102547304102412     
    286.2938003556570        0.6838204988365642        0.9006957089621104     
    285.9073683100782        0.6649512178664756        0.8911440113525845     
    285.5188909702327        0.6464751256820652        0.8816871329994941     
    285.1192893854838        0.6277983000829204        0.8723518982651232     
    284.7176818867101        0.6095436325195228        0.8631086218652259     
    284.2779897288794        0.5895000657624198        0.8540647708789280     
    283.8364233760649        0.5700143355097860        0.8451074007466638     
    283.3840278225319        0.5505642003336484        0.8362621200824832     
    282.8494485041059        0.5272810212846347        0.8277358572067062     
    282.3133130337449        0.5049109685780594        0.8192850610349551     
    281.7756237835377        0.4834216760581986        0.8109090999595029     
    281.2187136308625        0.4619357507887317        0.8026577773032617     
    280.6867274620471        0.4425546311006381        0.7944036623598385     
    280.1531638970493        0.4239257747481082        0.7862233426599742     
    279.6005405045062        0.4052882527902985        0.7781648528858310     
    279.0812526042587        0.3888183353821759        0.7700811691138462     
    278.4648411079485        0.3692907961927348        0.7623323422062077     
    277.8038827411572        0.3491368937205086        0.7547654787950228     
    277.1417837413831        0.3300517675685469        0.7472589319892405     
    276.4699558385488        0.3117030318083959        0.7398352779985683     
    275.7970200504041        0.2943477674320216        0.7324704128141959     
    275.1229775212685        0.2779337803016469        0.7251639453753789     
    274.4478293920471        0.2624115247418802        0.7179154871979165     
    273.7631090269861        0.2475166318568684        0.7107466357962663     
    273.1110652161774        0.2342652876225611        0.7035473877780448     
    272.5082714527131        0.2228637998141161        0.6962781471754149     
    271.9041058078915        0.2119919652308636        0.6890715421395707     
    271.6491281736033        0.2090898957103602        0.6810470739711953     
    271.1420293993023        0.2009085299998873        0.6737344914857260     
    270.5420567604472        0.1912009365650775        0.6667156683950179     
    269.9406918506893        0.1819403856192001        0.6597579312237877     
    269.3379365335090        0.1731074687501103        0.6528608014265593     
    268.7255963142966        0.1645436151602836        0.6460435086030647     
    268.1037327188147        0.1600000000000000        0.6393047039568076     
    267.4805436424469        0.1600000000000000        0.6326238224034243     
    266.8560307598823        0.1600000000000000        0.6260004242714480     
    266.2221188580712        0.1600000000000000        0.6194528662480088     
    265.5788689618569        0.1600000000000000        0.6129798792011273     
    264.9343593185875        0.1600000000000000        0.6065618640835542     
    264.2885914257266        0.1600000000000000        0.6001984168617190     
    263.6336087462312        0.1600000000000000        0.5939070636097012     
    262.9774002420890        0.1600000000000000        0.5876688912091225     
    262.3199673217555        0.1600000000000000        0.5814835151633467     
    261.6534421127312        0.1600000000000000        0.5753678575403014     
    260.9778848648271        0.1600000000000000        0.5693207686744767     
    260.3089762525495        0.1600000000000000        0.5633073175012350     
    259.6233162878010        0.1600000000000000        0.5573780153331546     
    258.9442778675530        0.1600000000000000        0.5514814864209824     
    258.2563594762279        0.1600000000000000        0.5456507590480206     
    257.6288552516643        0.1600000000000000        0.5397396718203187     
    257.0229818576733        0.1600000000000000        0.5338334128498313     
    256.4080294413739        0.1600000000000000        0.5279966768979322     
    255.7840582548341        0.1600000000000000        0.5222282917259126     
    255.1662798705100        0.1600000000000000        0.5164964389315726     
    254.5319981849248        0.1600000000000000        0.5108468593547970     
    253.8963661376447        0.1600000000000000        0.5052478043249716     
    253.2593854506084        0.1600000000000000        0.4996988793379387     
    252.6135984098757        0.1600000000000000        0.4942142862510941     
    251.9664949565847        0.1600000000000000        0.4887785419909243     
    251.3106752125899        0.1600000000000000        0.4834055075565324     
    250.6461984314630        0.1600000000000000        0.4780941463581427     
    249.9878111673873        0.1600000000000000        0.4728156623827861     
    249.3135127110329        0.1600000000000000        0.4676115128809167     
    248.6452777097507        0.1600000000000000        0.4624394086091887     
    247.9612781962457        0.1600000000000000        0.4573398022337801     
    247.2833159819885        0.1600000000000000        0.4522714395466867     
    246.5969353336135        0.1600000000000000        0.4472607346626793     
    245.9021947542256        0.1600000000000000        0.4423067490119886     
    245.2062952311828        0.1600000000000000        0.4373958218526404     
    244.5021239489411        0.1600000000000000        0.4325402316797045     
    243.8464228177035        0.1600000000000000        0.4276396549328401     
    243.2035101562650        0.1600000000000000        0.4227593058055832     
    242.5521886434440        0.1600000000000000        0.4179363533999879     
    241.8995167863185        0.1600000000000000        0.4131578911563659     
    241.2385246770844        0.1600000000000000        0.4084353697566481     
    240.5762138027817        0.1600000000000000        0.4037562145590233     
    239.9056708701831        0.1600000000000000        0.3991315909775053     
    239.2338405012449        0.1600000000000000        0.3945492473771000     
    238.5538658737074        0.1600000000000000        0.3900200710654035     
    237.8726348951976        0.1600000000000000        0.3855321248001752     
    237.1833470747868        0.1600000000000000        0.3810960247246966     
    236.5402501828367        0.1600000000000000        0.3766246270322433     
    235.9227162499828        0.1600000000000000        0.3721533170992779     
    235.3775464389630        0.1600000000000000        0.3676104432857336     
    234.5897189698534        0.1600000000000000        0.3634882073116544     
    233.8210469562954        0.1600000000000000        0.3593690185548979     
    233.0448292122683        0.1600000000000000        0.3552945814302490     
    232.2677277744820        0.1600000000000000        0.3512542136857281     
    231.4765878126806        0.1600000000000000        0.3472674431649149     
    230.6649713794084        0.1600000000000000        0.3433428817243095     
    229.7939141407275        0.1600000000000000        0.3395367769893946     
    228.9028809875759        0.1600000000000000        0.3357878437971146     
    228.0114372611108        0.1600000000000000        0.3320664132431559     
    227.1260212962541        0.1600000000000000        0.3283630666785231     
    */

namespace
{
  std::vector<double> CAMP2EX_aerosol_profile_factor = {
    1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000, 1.000000000000000,0.9610532350560985,0.9152870540745627,0.8715965240057739,0.8290917809871519,0.7885691895200798,0.7492219896966925,0.7268910657539789,0.7050930914993466,0.6838204988365642,0.6649512178664756,0.6464751256820652,0.6277983000829204,0.6095436325195228,0.5895000657624198,0.5700143355097860,0.5505642003336484,0.5272810212846347,0.5049109685780594,0.4834216760581986,0.4619357507887317,0.4425546311006381,0.4239257747481082,0.4052882527902985,0.3888183353821759,0.3692907961927348,0.3491368937205086,0.3300517675685469,0.3117030318083959,0.2943477674320216,0.2779337803016469,0.2624115247418802,0.2475166318568684,0.2342652876225611,0.2228637998141161,0.2119919652308636,0.2090898957103602,0.2009085299998873,0.1912009365650775,0.1819403856192001,0.1731074687501103,0.1645436151602836,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000,0.1600000000000000
  };

  std::vector<double> CAMP2EX_aerosol_profile_density = {
    1.167964615529032, 1.150449720531184, 1.141087409674031, 1.131770836523778, 1.122499926975512, 1.111024151109086, 1.100131508534285, 1.089296873796241, 1.078556570625417, 1.067874644671132, 1.057078088564207, 1.046381212479095, 1.035817012238898, 1.025349568943642, 1.015408695017796, 1.005614897689672,0.9958027537816577,0.9860719172834188,0.9764534662322681,0.9669141140649613,0.9574532680076016,0.9481009930467961,0.9388251106651607,0.9296550366246522,0.9199075280740460,0.9102547304102412,0.9006957089621104,0.8911440113525845,0.8816871329994941,0.8723518982651232,0.8631086218652259,0.8540647708789280,0.8451074007466638,0.8362621200824832,0.8277358572067062,0.8192850610349551,0.8109090999595029,0.8026577773032617,0.7944036623598385,0.7862233426599742,0.7781648528858310,0.7700811691138462,0.7623323422062077,0.7547654787950228,0.7472589319892405,0.7398352779985683,0.7324704128141959,0.7251639453753789,0.7179154871979165,0.7107466357962663,0.7035473877780448,0.6962781471754149,0.6890715421395707,0.6810470739711953,0.6737344914857260,0.6667156683950179,0.6597579312237877,0.6528608014265593,0.6460435086030647,0.6393047039568076,0.6326238224034243,0.6260004242714480,0.6194528662480088,0.6129798792011273,0.6065618640835542,0.6001984168617190,0.5939070636097012,0.5876688912091225,0.5814835151633467,0.5753678575403014,0.5693207686744767,0.5633073175012350,0.5573780153331546,0.5514814864209824,0.5456507590480206,0.5397396718203187,0.5338334128498313,0.5279966768979322,0.5222282917259126,0.5164964389315726,0.5108468593547970,0.5052478043249716,0.4996988793379387,0.4942142862510941,0.4887785419909243,0.4834055075565324,0.4780941463581427,0.4728156623827861,0.4676115128809167,0.4624394086091887,0.4573398022337801,0.4522714395466867,0.4472607346626793,0.4423067490119886,0.4373958218526404,0.4325402316797045,0.4276396549328401,0.4227593058055832,0.4179363533999879,0.4131578911563659,0.4084353697566481,0.4037562145590233,0.3991315909775053,0.3945492473771000,0.3900200710654035,0.3855321248001752,0.3810960247246966,0.3766246270322433,0.3721533170992779,0.3676104432857336,0.3634882073116544,0.3593690185548979,0.3552945814302490,0.3512542136857281,0.3472674431649149,0.3433428817243095,0.3395367769893946,0.3357878437971146,0.3320664132431559,0.3283630666785231
  };
};
