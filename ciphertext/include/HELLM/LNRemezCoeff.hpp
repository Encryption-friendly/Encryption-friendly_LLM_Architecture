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
namespace HELLM::LayerNorm {

/* input range

*/

// input range [0, 1]
static const HEaaN::Math::ChebyshevCoefficients
    LAYER_INVERSE_SQRT_COEFFS_127_16(
        {3.9359587065628147684037685394287109375000,
         -5.1385534462947362044360488653182983398438,
         4.2535060641399837777498760260641574859619,
         -3.6382258778900791185151319950819015502930,
         3.1679572070094366154080489650368690490723,
         -2.7778237066418398626410635188221931457520,
         2.4382166798272351115883793681859970092773,
         -2.1330843351777559746551560238003730773926,
         1.8526928674257305829087272286415100097656,
         -1.5906436040046969537797849625349044799805,
         1.3424577483493465024366741999983787536621,
         -1.1048324266816962335724383592605590820312,
         0.8752171434696549567888723686337471008301,
         -0.6515560308352519314212258905172348022461,
         0.4321211561568816250655800104141235351562,
         -0.2153978368579032576235476881265640258789,
         1.4074671664789093483705073595046997070312,
         -2.6041437986320943309692665934562683105469,
         2.4013587684083859130623750388622283935547,
         -2.2055517285780297243036329746246337890625,
         2.0158386019751333151361905038356781005859,
         -1.8314497979609996036742813885211944580078,
         1.6517078820573942721239291131496429443359,
         -1.4760100849193804606329649686813354492188,
         1.3038143980671748067834414541721343994141,
         -1.1346283588668484298978000879287719726562,
         0.9679998701287786389002576470375061035156,
         -0.8035095687166631250875070691108703613281,
         0.6407643765887769404798746109008789062500,
         -0.4793919521639509184751659631729125976562,
         0.3190358203648884227732196450233459472656,
         -0.1593510028342279838398098945617675781250,
         0.9400984998711692242068238556385040283203,
         -1.7519438704805452289292588829994201660156,
         1.6258980198417702922597527503967285156250,
         -1.5018665692487047635950148105621337890625,
         1.3796675979942847334314137697219848632812,
         -1.2591287676057163480436429381370544433594,
         1.1400861106269530864665284752845764160156,
         -1.0223829568053588445764034986495971679688,
         0.9058689742114438558928668498992919921875,
         -0.7903993064824135217349976301193237304688,
         0.6758337902729181223548948764801025390625,
         -0.5620362393519826582632958889007568359375,
         0.4488737836331893049646168947219848632812,
         -0.3362162529169836489018052816390991210938,
         0.2239355963070011057425290346145629882812,
         -0.1119053291386080672964453697204589843750,
         0.8285970861938949383329600095748901367188,
         -1.5461029735361080383881926536560058593750,
         1.4365175589828140800818800926208496093750,
         -1.3283198865165104507468640804290771484375,
         1.2213959003956915694288909435272216796875,
         -1.1156351203408121364191174507141113281250,
         1.0109302624869087594561278820037841796875,
         -0.9071768882022297475486993789672851562500,
         0.8042730771867354633286595344543457031250,
         -0.7021191215844737598672509193420410156250,
         0.6006172382312797708436846733093261718750,
         -0.4996712964057223871350288391113281250000,
         0.3991865587104257429018616676330566406250,
         -0.2990694328327663242816925048828125000000,
         0.1992272322167991660535335540771484375000,
         -0.0995679436455247923731803894042968750000,
         0.3381501901517367514315992593765258789062,
         -0.4697765963992424076423048973083496093750,
         0.5654107095181473141565220430493354797363,
         -0.5226923917711019385023973882198333740234,
         0.4804978252034857177932281047105789184570,
         -0.4387845433894312918710056692361831665039,
         0.3975106239139449826325289905071258544922,
         -0.3566346339344477200938854366540908813477,
         0.3161155782468085817527025938034057617188,
         -0.2759128495821414617239497601985931396484,
         0.2359861808483287859417032450437545776367,
         -0.1962955990593400201760232448577880859375,
         0.1568013807187185193470213562250137329102,
         -0.1174640084171869602869264781475067138672,
         0.0782441284400192671455442905426025390625,
         -0.0391025091558958592941053211688995361328,
         0.2880644613260301412083208560943603515625,
         -0.5373197219610119645949453115463256835938,
         0.4990582720508882630383595824241638183594,
         -0.4613061553263833047822117805480957031250,
         0.4240253651976217952324077486991882324219,
         -0.3871782702337895898381248116493225097656,
         0.3507275798353930440498515963554382324219,
         -0.3146363112964536412619054317474365234375,
         0.2788677580913372366921976208686828613281,
         -0.2433854592645730008371174335479736328125,
         0.2081531697917853307444602251052856445312,
         -0.1731348317866832076106220483779907226562,
         0.1382945464429212734103202819824218750000,
         -0.1035965465898698312230408191680908203125,
         0.0690051697642957151401787996292114257812,
         -0.0344848316781281027942895889282226562500,
         0.1779637883316809165989980101585388183594,
         -0.0094531028639721625950187444686889648438,
         0.2664505723101342482550535351037979125977,
         -0.2459608073738763778237625956535339355469,
         0.2257965046061372049734927713871002197266,
         -0.2059320239847011180245317518711090087891,
         0.1863418741170335124479606747627258300781,
         -0.1670006983514440435101278126239776611328,
         0.1478832612947371671907603740692138671875,
         -0.1289644356909320777049288153648376464844,
         0.1102191896192152853473089635372161865234,
         -0.0916225739747460465878248214721679687500,
         0.0731497101934337479178793728351593017578,
         -0.0547757781830569001613184809684753417969,
         0.0364760044258218840695917606353759765625,
         -0.0182256502159816591301932930946350097656,
         0.1501877795299151330254971981048583984375,
         -0.2825462020050508726853877305984497070312,
         0.2654850042931684583891183137893676757812,
         -0.2491680816660846176091581583023071289062,
         0.2335720817484343569958582520484924316406,
         -0.2186741736509247857611626386642456054688,
         0.2044520379820369271328672766685485839844,
         -0.1908838572458080307114869356155395507812,
         0.1779483065979547973256558179855346679688,
         -0.1656245449373727751662954688072204589844,
         0.1538922063114114280324429273605346679688,
         -0.1427313916126422554953023791313171386719,
         0.1321226605493848182959482073783874511719,
         -0.1220470238699817855376750230789184570312,
         0.1124859358259300279314629733562469482422,
         -0.6590982859306677710264921188354492187500},
        16);

static const HEaaN::Math::ChebyshevCoefficients LAYER_INVERSE_SQRT_127 =
    LAYER_INVERSE_SQRT_COEFFS_127_16;

} // namespace HELLM::LayerNorm
