////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2024 Crypto Lab Inc. All rights reserved.               //
//                                                                            //
// This source code is protected under international copyright law.           //
// All rights reserved and protected by the copyright holders.                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN-math/tools/PolynomialEvaluator.hpp"
#include "HEaaN/HEaaN.hpp"
#include <vector>
namespace HELLM::Tanh {

/* input range

*/

// input range [-5, 5]
static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS_63_8(
    {0.0000000000000000000000000000000000000000000000033200580,
     1.2628448865139472179208723900956101715564727783203125000,
     0.0000000000000000000000000000000000000000000000050364594,
     -0.3815488589895728033241084631299600005149841308593750000,
     0.0000000000000000000000000000000000000000000000033864224,
     0.1993220141048638871517084680817788466811180114746093750,
     0.0000000000000000000000000000000000000000000000017018977,
     -0.1378259695911224214093238060740986838936805725097656250,
     0.0000000000000000000000000000000000000000000000036124790,
     0.0962956229756924964036102210229728370904922485351562500,
     0.0000000000000000000000000000000000000000000000054855884,
     -0.0533035644876863889951579267290071584284305572509765625,
     0.0000000000000000000000000000000000000000000000036891473,
     0.0317226845053290712472815471301146317273378372192382812,
     0.0000000000000000000000000000000000000000000000018542491,
     -0.0227332051167677892666851846570352790877223014831542969,
     0.0000000000000000000000000000000000000000000000046836614,
     0.0081207012696459558720452065472272806800901889801025391,
     0.0000000000000000000000000000000000000000000000071038474,
     -0.0045143021104947986255062630789325339719653129577636719,
     0.0000000000000000000000000000000000000000000000047733511,
     0.0026905084650483655772656987892332836054265499114990234,
     0.0000000000000000000000000000000000000000000000023979393,
     -0.0019289656437958616064243244636600138619542121887207031,
     0.0000000000000000000000000000000000000000000000050513871,
     0.0013589744334692177574197557987645268440246582031250000,
     0.0000000000000000000000000000000000000000000000076585807,
     -0.0007554991771371866171591591410106047987937927246093750,
     0.0000000000000000000000000000000000000000000000051446335,
     0.0004502840775584271426623672596178948879241943359375000,
     0.0000000000000000000000000000000000000000000000025840110,
     -0.0003228342008795292983336366887670010328292846679687500,
     0.0000000000000000000000000000000000000000000000035182868,
     0.0000576727306498897384452392844700341356656281277537346,
     0.0000000000000000000000000000000000000000000000053466728,
     -0.0000320610169324262220778842813473374917521141469478607,
     0.0000000000000000000000000000000000000000000000036012664,
     0.0000191086504254462438007200475453828403260558843612671,
     0.0000000000000000000000000000000000000000000000018117755,
     -0.0000137000768491052121907103344966571967233903706073761,
     0.0000000000000000000000000000000000000000000000039017526,
     0.0000096518484984432252604680257945801713503897190093994,
     0.0000000000000000000000000000000000000000000000059420331,
     -0.0000053657888619449648631709592905281169805675745010376,
     0.0000000000000000000000000000000000000000000000040043908,
     0.0000031980577616464236911930640872014919295907020568848,
     0.0000000000000000000000000000000000000000000000020151982,
     -0.0000022928692599982758437704433163162320852279663085938,
     0.0000000000000000000000000000000000000000000000021486851,
     0.0000008177405051526824477509114430517911387141793966293,
     0.0000000000000000000000000000000000000000000000032692101,
     -0.0000004522030278120798760244092306948004988953471183777,
     0.0000000000000000000000000000000000000000000000022089231,
     0.0000002695170160864172881343137078147265128791332244873,
     0.0000000000000000000000000000000000000000000000011134443,
     -0.0000001932320574865109899520554392893245676532387733459,
     0.0000000000000000000000000000000000000000000000010997578,
     0.0000001353171628656977404370209683293069247156381607056,
     0.0000000000000000000000000000000000000000000000012204437,
     -0.0000000729084482704346614756474309615441597998142242432,
     0.0000000000000000000000000000000000000000000000005264304,
     0.0000000392828353524329182411278793551900889724493026733,
     0.0000000000000000000000000000000000000000000000001297155,
     -0.0000000298231959693670382882668690172067726962268352509},
    8);

// input range: (-12,12)
static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS12_127_16(
    {0.0000000000000000000000000000000000000000000000211372939,
     1.2799740803601333816175156243843957781791687011718750000,
     0.0000000000000000000000000000000000000000000000372179760,
     -0.4247224642992032395127921517996583133935928344726562500,
     0.0000000000000000000000000000000000000000000000320787374,
     0.2498031814061013322625370847163139842450618743896484375,
     0.0000000000000000000000000000000000000000000000268580858,
     -0.1735265377161906708014527112027280963957309722900390625,
     0.0000000000000000000000000000000000000000000000215691208,
     0.1315502440568506459239728201282559894025325775146484375,
     0.0000000000000000000000000000000000000000000000162251978,
     -0.1065471143925520447481858354876749217510223388671875000,
     0.0000000000000000000000000000000000000000000000108398770,
     0.0920649972253788362319681937151472084224224090576171875,
     0.0000000000000000000000000000000000000000000000054268723,
     -0.0853574061378212234352247378410538658499717712402343750,
     0.0000000000000000000000000000000000000000000000230080225,
     0.0742507210609903434139766886801226064562797546386718750,
     0.0000000000000000000000000000000000000000000000405294924,
     -0.0576581767482963955528951771611900767311453819274902344,
     0.0000000000000000000000000000000000000000000000349376438,
     0.0452595758337837883455989640424377284944057464599609375,
     0.0000000000000000000000000000000000000000000000292549082,
     -0.0360741216214550894370027833701897179707884788513183594,
     0.0000000000000000000000000000000000000000000000234959879,
     0.0294124079382058270515365450137323932722210884094238281,
     0.0000000000000000000000000000000000000000000000176758289,
     -0.0247903754067307739716863324019868741743266582489013672,
     0.0000000000000000000000000000000000000000000000118095714,
     0.0218786521042009449167231593946780776605010032653808594,
     0.0000000000000000000000000000000000000000000000059125010,
     -0.0204720629012225437126648586172450450249016284942626953,
     0.0000000000000000000000000000000000000000000000298150643,
     0.0092213428277927722565010526523110456764698028564453125,
     0.0000000000000000000000000000000000000000000000524872469,
     -0.0071971066128353987367827215848592459224164485931396484,
     0.0000000000000000000000000000000000000000000000452205190,
     0.0056666612628038862120583019077457720413804054260253906,
     0.0000000000000000000000000000000000000000000000378472724,
     -0.0045248392016188215825067686637339647859334945678710938,
     0.0000000000000000000000000000000000000000000000303850055,
     0.0036932546593735765538824011855467688292264938354492188,
     0.0000000000000000000000000000000000000000000000228513569,
     -0.0031148530590549838859004694313625805079936981201171875,
     0.0000000000000000000000000000000000000000000000152640760,
     0.0027499643389191225084644543130707461386919021606445312,
     0.0000000000000000000000000000000000000000000000076409949,
     -0.0025735673210430243251778392732376232743263244628906250,
     0.0000000000000000000000000000000000000000000000321511605,
     0.0022496313682611734918737056432291865348815917968750000,
     0.0000000000000000000000000000000000000000000000565878477,
     -0.0017559515209168430338593225314980372786521911621093750,
     0.0000000000000000000000000000000000000000000000487444202,
     0.0013826249028707771060453524114564061164855957031250000,
     0.0000000000000000000000000000000000000000000000407901921,
     -0.0011040629284352343475461566413287073373794555664062500,
     0.0000000000000000000000000000000000000000000000327434433,
     0.0009011727115701972934402874670922756195068359375000000,
     0.0000000000000000000000000000000000000000000000246225572,
     -0.0007600481212479692771921691019088029861450195312500000,
     0.0000000000000000000000000000000000000000000000164459983,
     0.0006710165130876521999425676767714321613311767578125000,
     0.0000000000000000000000000000000000000000000000082322911,
     -0.0006279757043126110716002585832029581069946289062500000,
     0.0000000000000000000000000000000000000000000000224139946,
     0.0001415383584268367937394894617852969531668350100517273,
     0.0000000000000000000000000000000000000000000000395073616,
     -0.0001104304908313226462127087934561586735071614384651184,
     0.0000000000000000000000000000000000000000000000340903843,
     0.0000869523366368273042947834028382203541696071624755859,
     0.0000000000000000000000000000000000000000000000285697257,
     -0.0000694338027543315791656475255422265036031603813171387,
     0.0000000000000000000000000000000000000000000000229617860,
     0.0000566741872047120658767438428071727685164660215377808,
     0.0000000000000000000000000000000000000000000000172834452,
     -0.0000477989594487198462408050758654098899569362401962280,
     0.0000000000000000000000000000000000000000000000115519671,
     0.0000421998208385886489080185413058643462136387825012207,
     0.0000000000000000000000000000000000000000000000057849039,
     -0.0000394930124069238323858410666389318066649138927459717,
     0.0000000000000000000000000000000000000000000000248769560,
     0.0000345220613174835808112383972456882474943995475769043,
     0.0000000000000000000000000000000000000000000000438897247,
     -0.0000269462639410000467932171375196048757061362266540527,
     0.0000000000000000000000000000000000000000000000378851825,
     0.0000212173343887064186352731098850199487060308456420898,
     0.0000000000000000000000000000000000000000000000317591444,
     -0.0000169426180139185117434763583332824055105447769165039,
     0.0000000000000000000000000000000000000000000000255310212,
     0.0000138291303727507142173891452330281026661396026611328,
     0.0000000000000000000000000000000000000000000000192206825,
     -0.0000116634763853359912921803243079921230673789978027344,
     0.0000000000000000000000000000000000000000000000128483634,
     0.0000102972244669270740491384685810771770775318145751953,
     0.0000000000000000000000000000000000000000000000064345724,
     -0.0000096367331870808592420019067503744736313819885253906,
     0.0000000000000000000000000000000000000000000000136979901,
     0.0000043983662990005386456138225526046880986541509628296,
     0.0000000000000000000000000000000000000000000000241709342,
     -0.0000033380441477831677282717137700274179223924875259399,
     0.0000000000000000000000000000000000000000000000208988677,
     0.0000026283569055443600781529767118627205491065979003906,
     0.0000000000000000000000000000000000000000000000175449074,
     -0.0000020988144055541363419692402203509118407964706420898,
     0.0000000000000000000000000000000000000000000000141213633,
     0.0000017131223770581148480030009295660420320928096771240,
     0.0000000000000000000000000000000000000000000000106412727,
     -0.0000014448459051750843495920406667210045270621776580811,
     0.0000000000000000000000000000000000000000000000071182577,
     0.0000012755976104093097456271266310068313032388687133789,
     0.0000000000000000000000000000000000000000000000035663811,
     -0.0000011937773975170470897033681012544548138976097106934,
     0.0000000000000000000000000000000000000000000000069889532,
     0.0000010388341190117889034638665179954841732978820800781,
     0.0000000000000000000000000000000000000000000000106253520,
     -0.0000008001481300087126546927152048738207668066024780273,
     0.0000000000000000000000000000000000000000000000077143058,
     0.0000006163034292376232126109414366510463878512382507324,
     0.0000000000000000000000000000000000000000000000052571711,
     -0.0000004746994995618800117220814627216896042227745056152,
     0.0000000000000000000000000000000000000000000000032647444,
     0.0000003656309606504815068461766713880933821201324462891,
     0.0000000000000000000000000000000000000000000000017460531,
     -0.0000002816223726997968412888440070673823356628417968750,
     0.0000000000000000000000000000000000000000000000007081902,
     0.0000002169158778678355325997273439497803337872028350830,
     0.0000000000000000000000000000000000000000000000001561449,
     -0.0000004107745028843028112230939541404950432479381561279},
    16);

// input range: (-16,16)
static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS16_127_16(
    {0.0000000000000000000000000000000000000000000000193376271,
     1.2939363021610945647665857904939912259578704833984375000,
     0.0000000000000000000000000000000000000000000000340486883,
     -0.4419505078146984677900377391779329627752304077148437500,
     0.0000000000000000000000000000000000000000000000293472517,
     0.2701212807274101512788888612703885883092880249023437500,
     0.0000000000000000000000000000000000000000000000245712577,
     -0.1965703722937938713322125749982660636305809020996093750,
     0.0000000000000000000000000000000000000000000000197327016,
     0.1568324445201482453260410920847789384424686431884765625,
     0.0000000000000000000000000000000000000000000000148438136,
     -0.1335179979820371454213301376512390561401844024658203125,
     0.0000000000000000000000000000000000000000000000099170121,
     0.1201565919511462798840994992133346386253833770751953125,
     0.0000000000000000000000000000000000000000000000049648563,
     -0.1140056961016684133891629926438326947391033172607421875,
     0.0000000000000000000000000000000000000000000000210508428,
     0.1027689473976250017495104316367360297590494155883789062,
     0.0000000000000000000000000000000000000000000000370821583,
     -0.0854682492726201309407585426924924831837415695190429688,
     0.0000000000000000000000000000000000000000000000319661711,
     0.0721127291755600252631808189107687212526798248291015625,
     0.0000000000000000000000000000000000000000000000267669221,
     -0.0618913634271127111041721491346834227442741394042968750,
     0.0000000000000000000000000000000000000000000000214978787,
     0.0542510439419534029603653380036121234297752380371093750,
     0.0000000000000000000000000000000000000000000000161727325,
     -0.0488111275821316858980480901664122939109802246093750000,
     0.0000000000000000000000000000000000000000000000108053534,
     0.0453155829433183610088775594704202376306056976318359375,
     0.0000000000000000000000000000000000000000000000054097451,
     -0.0436058136391647144236571875808294862508773803710937500,
     0.0000000000000000000000000000000000000000000000272892838,
     0.0215584213612414199445765916607342660427093505859375000,
     0.0000000000000000000000000000000000000000000000480411648,
     -0.0181382221348379557568364361941348761320114135742187500,
     0.0000000000000000000000000000000000000000000000413902611,
     0.0154230341153649419538851361721754074096679687500000000,
     0.0000000000000000000000000000000000000000000000346417356,
     -0.0133056546938444819616620407032314687967300415039062500,
     0.0000000000000000000000000000000000000000000000278116247,
     0.0117030462200765628111298610747326165437698364257812500,
     0.0000000000000000000000000000000000000000000000209160934,
     -0.0105526428147416639902189672284293919801712036132812500,
     0.0000000000000000000000000000000000000000000000139714087,
     0.0098096659804760566458980974857695400714874267578125000,
     0.0000000000000000000000000000000000000000000000069939133,
     -0.0094452495559875782049630288383923470973968505859375000,
     0.0000000000000000000000000000000000000000000000294303814,
     0.0085990556584651933746954455273225903511047363281250000,
     0.0000000000000000000000000000000000000000000000517995037,
     -0.0072383838953093970758345676586031913757324218750000000,
     0.0000000000000000000000000000000000000000000000446200522,
     0.0061568839234356076417498115915805101394653320312500000,
     0.0000000000000000000000000000000000000000000000373390463,
     -0.0053128124103798413102595077361911535263061523437500000,
     0.0000000000000000000000000000000000000000000000299732403,
     0.0046736008915164717336665489710867404937744140625000000,
     0.0000000000000000000000000000000000000000000000225394834,
     -0.0042145908310473600977275054901838302612304687500000000,
     0.0000000000000000000000000000000000000000000000150546991,
     0.0039180777783656139945378527045249938964843750000000000,
     0.0000000000000000000000000000000000000000000000075358662,
     -0.0037726259702548503582875127904117107391357421875000000,
     0.0000000000000000000000000000000000000000000000204901659,
     0.0009389508697928983238512046227697283029556274414062500,
     0.0000000000000000000000000000000000000000000000361152860,
     -0.0007869700887687992862939978522263118065893650054931641,
     0.0000000000000000000000000000000000000000000000311636655,
     0.0006693981742855979744089012228869250975549221038818359,
     0.0000000000000000000000000000000000000000000000261171506,
     -0.0005776339564808122273298351956327678635716438293457031,
     0.0000000000000000000000000000000000000000000000209907492,
     0.0005081395076336897576041451429773587733507156372070312,
     0.0000000000000000000000000000000000000000000000157999091,
     -0.0004582355268057461405994956749054836109280586242675781,
     0.0000000000000000000000000000000000000000000000105604312,
     0.0004259980195420454140986521451850421726703643798828125,
     0.0000000000000000000000000000000000000000000000052883808,
     -0.0004101841082419510620127311995020136237144470214843750,
     0.0000000000000000000000000000000000000000000000227444874,
     0.0003734433560618477532244696703855879604816436767578125,
     0.0000000000000000000000000000000000000000000000401279950,
     -0.0003143592844828402288470670100650750100612640380859375,
     0.0000000000000000000000000000000000000000000000346384890,
     0.0002673947611569471582981805113377049565315246582031250,
     0.0000000000000000000000000000000000000000000000290377226,
     -0.0002307391517127888036498006840702146291732788085937500,
     0.0000000000000000000000000000000000000000000000233434688,
     0.0002029792647569861330225648998748511075973510742187500,
     0.0000000000000000000000000000000000000000000000175739222,
     -0.0001830448680477658740528568159788846969604492187500000,
     0.0000000000000000000000000000000000000000000000117476129,
     0.0001701674272022724032638052449328824877738952636718750,
     0.0000000000000000000000000000000000000000000000058833226,
     -0.0001638504760215209188345397706143558025360107421875000,
     0.0000000000000000000000000000000000000000000000124959853,
     0.0000859628670963896723833386204205453395843505859375000,
     0.0000000000000000000000000000000000000000000000220473226,
     -0.0000655132583450915295664174209377961233258247375488281,
     0.0000000000000000000000000000000000000000000000190630129,
     0.0000557257365208438382175870628998382017016410827636719,
     0.0000000000000000000000000000000000000000000000160038854,
     -0.0000480866169945276533681521868857089430093765258789062,
     0.0000000000000000000000000000000000000000000000128811801,
     0.0000423013876810485189849941889406181871891021728515625,
     0.0000000000000000000000000000000000000000000000097068050,
     -0.0000381470097988861889248823899833951145410537719726562,
     0.0000000000000000000000000000000000000000000000064932052,
     0.0000354633190196573799823909212136641144752502441406250,
     0.0000000000000000000000000000000000000000000000032532302,
     -0.0000341468506274145155998667178209871053695678710937500,
     0.0000000000000000000000000000000000000000000000063481081,
     0.0000308105310479755978292359941406175494194030761718750,
     0.0000000000000000000000000000000000000000000000096389883,
     -0.0000253257346768953617299757752334699034690856933593750,
     0.0000000000000000000000000000000000000000000000069864225,
     0.0000208173249577614516425683177658356726169586181640625,
     0.0000000000000000000000000000000000000000000000047498421,
     -0.0000171114885288425888631991256261244416236877441406250,
     0.0000000000000000000000000000000000000000000000029390930,
     0.0000140653537508417247892111845430918037891387939453125,
     0.0000000000000000000000000000000000000000000000015623949,
     -0.0000115614825563076228931436162383761256933212280273438,
     0.0000000000000000000000000000000000000000000000006261894,
     0.0000095033428429008859339433001878205686807632446289062,
     0.0000000000000000000000000000000000000000000000001349849,
     -0.0000240842765899529354101105127483606338500976562500000},
    16);

// input range: (-12,12)
static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS_63_8_WIDE(
    {0.0000000000000000000000000000000000000281090481899584310,
     1.3653703271795831764023887444636784493923187255859375000,
     0.0000000000000000000000000000000000000426293586989588820,
     -0.5167111463698396134347490260552149266004562377929687500,
     0.0000000000000000000000000000000000000286658565386692050,
     0.3562829201511577004168884741375222802162170410156250000,
     0.0000000000000000000000000000000000000144072689531010226,
     -0.3050137277783990796820035029668360948562622070312500000,
     0.0000000000000000000000000000000000000306055367859830918,
     0.2630438139267106123497796943411231040954589843750000000,
     0.0000000000000000000000000000000000000464823326981544534,
     -0.2130464298257583877216347900684922933578491210937500000,
     0.0000000000000000000000000000000000000312637413061308372,
     0.1840877946299706025001796660944819450378417968750000000,
     0.0000000000000000000000000000000000000157149598801132742,
     -0.1706753192632675109052797779440879821777343750000000000,
     0.0000000000000000000000000000000000000398480818813319286,
     0.0947007045649712608792469836771488189697265625000000000,
     0.0000000000000000000000000000000000000604482511556990116,
     -0.0795182071082987107502049184404313564300537109375000000,
     0.0000000000000000000000000000000000000406220521214565320,
     0.0700335108351310964280855841934680938720703125000000000,
     0.0000000000000000000000000000000000000204082391357488121,
     -0.0654711436854675810081971576437354087829589843750000000,
     0.0000000000000000000000000000000000000430238101044551178,
     0.0588109867460389068583026528358459472656250000000000000,
     0.0000000000000000000000000000000000000652398471162553643,
     -0.0495690873370762119520804844796657562255859375000000000,
     0.0000000000000000000000000000000000000438294538381325849,
     0.0437470069839349662288441322743892669677734375000000000,
     0.0000000000000000000000000000000000000220157749544244375,
     -0.0409344890692580065660877153277397155761718750000000000,
     0.0000000000000000000000000000000000000295541335067850841,
     0.0118753875839412437187547766370698809623718261718750000,
     0.0000000000000000000000000000000000000448867859551096639,
     -0.0097967474631486251013257060549221932888031005859375000,
     0.0000000000000000000000000000000000000302377682258181824,
     0.0086487996282729184827076096553355455398559570312500000,
     0.0000000000000000000000000000000000000152137124254814601,
     -0.0080938918400992321267040097154676914215087890625000000,
     0.0000000000000000000000000000000000000328067031755265175,
     0.0072748740671428890891547780483961105346679687500000000,
     0.0000000000000000000000000000000000000499749476716379100,
     -0.0061355530453237605570393498055636882781982421875000000,
     0.0000000000000000000000000000000000000336849635693247925,
     0.0054168046338745057255437131971120834350585937500000000,
     0.0000000000000000000000000000000000000169537984788038514,
     -0.0050693423947336668788921087980270385742187500000000000,
     0.0000000000000000000000000000000000000176512023323749213,
     0.0030778533978823574557281972374767065048217773437500000,
     0.0000000000000000000000000000000000000267924913905011849,
     -0.0021594594354357177223846520064398646354675292968750000,
     0.0000000000000000000000000000000000000181072769859234165,
     0.0019065002078040294009042554534971714019775390625000000,
     0.0000000000000000000000000000000000000091286234565467164,
     -0.0017842115750532627771463012322783470153808593750000000,
     0.0000000000000000000000000000000000000086344506735376035,
     0.0015526341056713022226176690310239791870117187500000000,
     0.0000000000000000000000000000000000000093015340598154049,
     -0.0011958954750568295821722131222486495971679687500000000,
     0.0000000000000000000000000000000000000037918577686702675,
     0.0009211224361953540551439800765365362167358398437500000,
     0.0000000000000000000000000000000000000008387063332890858,
     -0.0017443331520308191784351947717368602752685546875000000},
    8);

static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS_63 =
    TANH_COEFFS_63_8;
static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS12_127 =
    TANH_COEFFS12_127_16;

static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS16_127 =
    TANH_COEFFS16_127_16;

static const HEaaN::Math::ChebyshevCoefficients TANH_COEFFS_63_WIDE =
    TANH_COEFFS_63_8_WIDE;

} // namespace HELLM::Tanh