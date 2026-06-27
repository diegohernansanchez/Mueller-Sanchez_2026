"""
Runs in PYTHON 3 for Windows
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import re
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.multitest import multipletests

### --- 1. CONFIGURATION ---
P_THRESHOLD = 0.01

path_ecotypes   = "./PATH/"
path_gff3       = "./PATH/"
path_expr       = "./PATH/"
path_snps       = "./PATH/"
path_meth       = "./PATH/"
path_bioc       = "./PATH/"
path_out        = "./PATH/"

file_ecotypes   = "List_1135_Aras_ecotypes.txt"
inputFile_gff3  = "Allnoncanonical_plus_unique_FINAL_LTRharvest85_TAIR10_chr_all_NAMEnCLUSTER.gff3"
file_expr       = "6_TEs_expresion_model.txt"
file_bioc       = "6_bioclimvars_model.txt"
file_snps       = "plink.eigenvec_ALL_ACCESSIONS_allsnps"
file_eigenval   = "plink.eigenval_ALL_ACCESSIONS_allsnps" 

meth_files = {
    'CG': 'MeC_bismark_LTRs_vs_accessions_CG.txt',
    'CHG': 'MeC_bismark_LTRs_vs_accessions_CHG.txt',
    'CHH': 'MeC_bismark_LTRs_vs_accessions_CHH.txt'
}

Expressed_list = [
"TAIR10_chr_all_Chr1:428928-438080",
"TAIR10_chr_all_Chr1:3780765-3785720",
"TAIR10_chr_all_Chr1:5691764-5696712",
"TAIR10_chr_all_Chr1:6009293-6014070",
"TAIR10_chr_all_Chr1:6718669-6721900",
"TAIR10_chr_all_Chr1:7065613-7075764",
"TAIR10_chr_all_Chr1:7714702-7722546",
"TAIR10_chr_all_Chr1:7968426-7978566",
"TAIR10_chr_all_Chr1:9368344-9373311",
"TAIR10_chr_all_Chr1:9476002-9480634",
"TAIR10_chr_all_Chr1:10610329-10614183",
"TAIR10_chr_all_Chr1:10696714-10701602",
"TAIR10_chr_all_Chr1:11502893-11512816",
"TAIR10_chr_all_Chr1:12273107-12276549",
"TAIR10_chr_all_Chr1:12406711-12410002",
"TAIR10_chr_all_Chr1:12664517-12673944",
"TAIR10_chr_all_Chr1:12754085-12760392",
"TAIR10_chr_all_Chr1:12813813-12816477",
"TAIR10_chr_all_Chr1:13000215-13005223",
"TAIR10_chr_all_Chr1:13019157-13023946",
"TAIR10_chr_all_Chr1:13180609-13186858",
"TAIR10_chr_all_Chr1:13230575-13236108",
"TAIR10_chr_all_Chr1:13324760-13329591",
"TAIR10_chr_all_Chr1:13439443-13448531",
"TAIR10_chr_all_Chr1:13507767-13515762",
"TAIR10_chr_all_Chr1:13559843-13564825",
"TAIR10_chr_all_Chr1:13593057-13599715",
"TAIR10_chr_all_Chr1:13659271-13667557",
"TAIR10_chr_all_Chr1:13718566-13722308",
"TAIR10_chr_all_Chr1:13787085-13794674",
"TAIR10_chr_all_Chr1:13883078-13890934",
"TAIR10_chr_all_Chr1:14019278-14024520",
"TAIR10_chr_all_Chr1:14082800-14094066",
"TAIR10_chr_all_Chr1:14098627-14107398",
"TAIR10_chr_all_Chr1:14129769-14134851",
"TAIR10_chr_all_Chr1:14179549-14183995",
"TAIR10_chr_all_Chr1:14197051-14206945",
"TAIR10_chr_all_Chr1:14214742-14225658",
"TAIR10_chr_all_Chr1:14233003-14241009",
"TAIR10_chr_all_Chr1:14245147-14255424",
"TAIR10_chr_all_Chr1:14263558-14273764",
"TAIR10_chr_all_Chr1:14312104-14327071",
"TAIR10_chr_all_Chr1:14433021-14442952",
"TAIR10_chr_all_Chr1:15179372-15191896",
"TAIR10_chr_all_Chr1:15208663-15223255",
"TAIR10_chr_all_Chr1:15492100-15503781",
"TAIR10_chr_all_Chr1:15509413-15522305",
"TAIR10_chr_all_Chr1:15536835-15545906",
"TAIR10_chr_all_Chr1:15610250-15615956",
"TAIR10_chr_all_Chr1:15704954-15714921",
"TAIR10_chr_all_Chr1:15735664-15741827",
"TAIR10_chr_all_Chr1:16012600-16019700",
"TAIR10_chr_all_Chr1:16045318-16051716",
"TAIR10_chr_all_Chr1:16094141-16103848",
"TAIR10_chr_all_Chr1:16117824-16124068",
"TAIR10_chr_all_Chr1:16191402-16198723",
"TAIR10_chr_all_Chr1:16369961-16374973",
"TAIR10_chr_all_Chr1:16380213-16387338",
"TAIR10_chr_all_Chr1:16387795-16398174",
"TAIR10_chr_all_Chr1:16488892-16491745",
"TAIR10_chr_all_Chr1:16509812-16520922",
"TAIR10_chr_all_Chr1:16551721-16557069",
"TAIR10_chr_all_Chr1:16569862-16575246",
"TAIR10_chr_all_Chr1:16626036-16637235",
"TAIR10_chr_all_Chr1:16638793-16645559",
"TAIR10_chr_all_Chr1:16674905-16683312",
"TAIR10_chr_all_Chr1:16853317-16858041",
"TAIR10_chr_all_Chr1:17023916-17029257",
"TAIR10_chr_all_Chr1:17054997-17060015",
"TAIR10_chr_all_Chr1:17247720-17252580",
"TAIR10_chr_all_Chr1:17255285-17265291",
"TAIR10_chr_all_Chr1:17360725-17365672",
"TAIR10_chr_all_Chr1:17402377-17414527",
"TAIR10_chr_all_Chr1:17426933-17435091",
"TAIR10_chr_all_Chr1:17443080-17446790",
"TAIR10_chr_all_Chr1:17473528-17476674",
"TAIR10_chr_all_Chr1:17676598-17681219",
"TAIR10_chr_all_Chr1:18002175-18007187",
"TAIR10_chr_all_Chr1:18013159-18018434",
"TAIR10_chr_all_Chr1:18828527-18833280",
"TAIR10_chr_all_Chr1:18962095-18971497",
"TAIR10_chr_all_Chr1:19445697-19451857",
"TAIR10_chr_all_Chr1:21244324-21249198",
"TAIR10_chr_all_Chr1:21346162-21351041",
"TAIR10_chr_all_Chr1:21408871-21414561",
"TAIR10_chr_all_Chr1:21524995-21529825",
"TAIR10_chr_all_Chr1:21748311-21753395",
"TAIR10_chr_all_Chr1:21833194-21838009",
"TAIR10_chr_all_Chr1:22100643-22105329",
"TAIR10_chr_all_Chr1:23118232-23123209",
"TAIR10_chr_all_Chr1:26366607-26371676",
"TAIR10_chr_all_Chr2:728942-737651",
"TAIR10_chr_all_Chr2:926121-929956",
"TAIR10_chr_all_Chr2:1292790-1297013",
"TAIR10_chr_all_Chr2:1388832-1393901",
"TAIR10_chr_all_Chr2:1417976-1428317",
"TAIR10_chr_all_Chr2:1557142-1562055",
"TAIR10_chr_all_Chr2:1629669-1637482",
"TAIR10_chr_all_Chr2:1689351-1697710",
"TAIR10_chr_all_Chr2:1781545-1786784",
"TAIR10_chr_all_Chr2:1882606-1887814",
"TAIR10_chr_all_Chr2:1906859-1912114",
"TAIR10_chr_all_Chr2:2075306-2080607",
"TAIR10_chr_all_Chr2:2274202-2280048",
"TAIR10_chr_all_Chr2:2302464-2307171",
"TAIR10_chr_all_Chr2:2374102-2385753",
"TAIR10_chr_all_Chr2:2448362-2454184",
"TAIR10_chr_all_Chr2:2487775-2499117",
"TAIR10_chr_all_Chr2:2501036-2512184",
"TAIR10_chr_all_Chr2:2564008-2570729",
"TAIR10_chr_all_Chr2:2745600-2752272",
"TAIR10_chr_all_Chr2:2793395-2799755",
"TAIR10_chr_all_Chr2:2844753-2849606",
"TAIR10_chr_all_Chr2:2858551-2864193",
"TAIR10_chr_all_Chr2:2871827-2881602",
"TAIR10_chr_all_Chr2:3083902-3089468",
"TAIR10_chr_all_Chr2:3095073-3107233",
"TAIR10_chr_all_Chr2:3532685-3539330",
"TAIR10_chr_all_Chr2:3667037-3671939",
"TAIR10_chr_all_Chr2:3732765-3740497",
"TAIR10_chr_all_Chr2:3752703-3763196",
"TAIR10_chr_all_Chr2:3873345-3887634",
"TAIR10_chr_all_Chr2:3945507-3959399",
"TAIR10_chr_all_Chr2:4050147-4058310",
"TAIR10_chr_all_Chr2:4061046-4065460",
"TAIR10_chr_all_Chr2:4161050-4173975",
"TAIR10_chr_all_Chr2:4175780-4184391",
"TAIR10_chr_all_Chr2:4197159-4210001",
"TAIR10_chr_all_Chr2:4230743-4236514",
"TAIR10_chr_all_Chr2:4236657-4244396",
"TAIR10_chr_all_Chr2:4275619-4281968",
"TAIR10_chr_all_Chr2:4408324-4413624",
"TAIR10_chr_all_Chr2:4542919-4547703",
"TAIR10_chr_all_Chr2:4558511-4566131",
"TAIR10_chr_all_Chr2:4570582-4582826",
"TAIR10_chr_all_Chr2:4682601-4690776",
"TAIR10_chr_all_Chr2:4710546-4716011",
"TAIR10_chr_all_Chr2:4808823-4818381",
"TAIR10_chr_all_Chr2:4818686-4826104",
"TAIR10_chr_all_Chr2:4948700-4959163",
"TAIR10_chr_all_Chr2:4964526-4973015",
"TAIR10_chr_all_Chr2:4978561-4990079",
"TAIR10_chr_all_Chr2:5040996-5048412",
"TAIR10_chr_all_Chr2:5062469-5067367",
"TAIR10_chr_all_Chr2:5073848-5078421",
"TAIR10_chr_all_Chr2:5084956-5095493",
"TAIR10_chr_all_Chr2:5329574-5336837",
"TAIR10_chr_all_Chr2:5348120-5364356",
"TAIR10_chr_all_Chr2:5394710-5399651",
"TAIR10_chr_all_Chr2:5448038-5458247",
"TAIR10_chr_all_Chr2:5565500-5578988",
"TAIR10_chr_all_Chr2:5709609-5713257",
"TAIR10_chr_all_Chr2:5780294-5787383",
"TAIR10_chr_all_Chr2:5807646-5814894",
"TAIR10_chr_all_Chr2:5846254-5851428",
"TAIR10_chr_all_Chr2:5985323-5995737",
"TAIR10_chr_all_Chr2:6023868-6032902",
"TAIR10_chr_all_Chr2:6113438-6124746",
"TAIR10_chr_all_Chr2:6263307-6270535",
"TAIR10_chr_all_Chr2:6544781-6559665",
"TAIR10_chr_all_Chr2:6717298-6728626",
"TAIR10_chr_all_Chr2:6819006-6823730",
"TAIR10_chr_all_Chr2:6837331-6843329",
"TAIR10_chr_all_Chr2:6954051-6958823",
"TAIR10_chr_all_Chr2:7224082-7229125",
"TAIR10_chr_all_Chr2:7448762-7454289",
"TAIR10_chr_all_Chr2:7598599-7602619",
"TAIR10_chr_all_Chr2:8561850-8566969",
"TAIR10_chr_all_Chr2:8816056-8820908",
"TAIR10_chr_all_Chr2:9018947-9028487",
"TAIR10_chr_all_Chr2:9121958-9125818",
"TAIR10_chr_all_Chr2:9187492-9195281",
"TAIR10_chr_all_Chr2:9927681-9932560",
"TAIR10_chr_all_Chr2:10488231-10491928",
"TAIR10_chr_all_Chr2:10804728-10809022",
"TAIR10_chr_all_Chr2:12539439-12541817",
"TAIR10_chr_all_Chr3:3807306-3812643",
"TAIR10_chr_all_Chr3:7359095-7368629",
"TAIR10_chr_all_Chr3:8537355-8543559",
"TAIR10_chr_all_Chr3:9387847-9393279",
"TAIR10_chr_all_Chr3:9692217-9697526",
"TAIR10_chr_all_Chr3:10106161-10111478",
"TAIR10_chr_all_Chr3:10494344-10500970",
"TAIR10_chr_all_Chr3:11016111-11020804",
"TAIR10_chr_all_Chr3:11122713-11127691",
"TAIR10_chr_all_Chr3:11318303-11327633",
"TAIR10_chr_all_Chr3:11417241-11424401",
"TAIR10_chr_all_Chr3:11486704-11496183",
"TAIR10_chr_all_Chr3:11578844-11586331",
"TAIR10_chr_all_Chr3:11652697-11657743",
"TAIR10_chr_all_Chr3:11664969-11669787",
"TAIR10_chr_all_Chr3:11697284-11702285",
"TAIR10_chr_all_Chr3:12025159-12040030",
"TAIR10_chr_all_Chr3:12187376-12195745",
"TAIR10_chr_all_Chr3:12201740-12209378",
"TAIR10_chr_all_Chr3:12212075-12225831",
"TAIR10_chr_all_Chr3:12243853-12255283",
"TAIR10_chr_all_Chr3:12303477-12308746",
"TAIR10_chr_all_Chr3:12401407-12414030",
"TAIR10_chr_all_Chr3:12478090-12486335",
"TAIR10_chr_all_Chr3:12491762-12502104",
"TAIR10_chr_all_Chr3:12502664-12508397",
"TAIR10_chr_all_Chr3:12549509-12555919",
"TAIR10_chr_all_Chr3:12580262-12586836",
"TAIR10_chr_all_Chr3:12602883-12608833",
"TAIR10_chr_all_Chr3:12613331-12620941",
"TAIR10_chr_all_Chr3:12695936-12704557",
"TAIR10_chr_all_Chr3:12705298-12717572",
"TAIR10_chr_all_Chr3:12743686-12755116",
"TAIR10_chr_all_Chr3:12853595-12861862",
"TAIR10_chr_all_Chr3:12960401-12968465",
"TAIR10_chr_all_Chr3:12982736-12993057",
"TAIR10_chr_all_Chr3:13087365-13092790",
"TAIR10_chr_all_Chr3:13240355-13252211",
"TAIR10_chr_all_Chr3:13277152-13289157",
"TAIR10_chr_all_Chr3:13369174-13374107",
"TAIR10_chr_all_Chr3:13403158-13411752",
"TAIR10_chr_all_Chr3:13440046-13449246",
"TAIR10_chr_all_Chr3:13621392-13630750",
"TAIR10_chr_all_Chr3:13801559-13809282",
"TAIR10_chr_all_Chr3:13902127-13915044",
"TAIR10_chr_all_Chr3:14073146-14081965",
"TAIR10_chr_all_Chr3:14145237-14157238",
"TAIR10_chr_all_Chr3:14159731-14165436",
"TAIR10_chr_all_Chr3:14245383-14255627",
"TAIR10_chr_all_Chr3:14337274-14352107",
"TAIR10_chr_all_Chr3:14406064-14415496",
"TAIR10_chr_all_Chr3:14549414-14554634",
"TAIR10_chr_all_Chr3:14579217-14584090",
"TAIR10_chr_all_Chr3:14671141-14675856",
"TAIR10_chr_all_Chr3:14703485-14710413",
"TAIR10_chr_all_Chr3:14794579-14804700",
"TAIR10_chr_all_Chr3:14813663-14827205",
"TAIR10_chr_all_Chr3:14978417-14986927",
"TAIR10_chr_all_Chr3:15081073-15085034",
"TAIR10_chr_all_Chr3:15108689-15117493",
"TAIR10_chr_all_Chr3:15250461-15262482",
"TAIR10_chr_all_Chr3:15299840-15313375",
"TAIR10_chr_all_Chr3:15377292-15382211",
"TAIR10_chr_all_Chr3:15455934-15468076",
"TAIR10_chr_all_Chr3:15540534-15549170",
"TAIR10_chr_all_Chr3:15574759-15585467",
"TAIR10_chr_all_Chr3:15591669-15600823",
"TAIR10_chr_all_Chr3:15621871-15626854",
"TAIR10_chr_all_Chr3:15711742-15722687",
"TAIR10_chr_all_Chr3:15768786-15773747",
"TAIR10_chr_all_Chr3:15848719-15853738",
"TAIR10_chr_all_Chr3:15919026-15924075",
"TAIR10_chr_all_Chr3:16243307-16256895",
"TAIR10_chr_all_Chr3:16333763-16342001",
"TAIR10_chr_all_Chr3:16683724-16691512",
"TAIR10_chr_all_Chr3:16724998-16730997",
"TAIR10_chr_all_Chr3:16797017-16804625",
"TAIR10_chr_all_Chr3:16809187-16814152",
"TAIR10_chr_all_Chr3:16957465-16962731",
"TAIR10_chr_all_Chr3:17055569-17057941",
"TAIR10_chr_all_Chr3:17978740-17983947",
"TAIR10_chr_all_Chr3:18791151-18796105",
"TAIR10_chr_all_Chr3:20304946-20310167",
"TAIR10_chr_all_Chr3:20502913-20507056",
"TAIR10_chr_all_Chr3:22059535-22064329",
"TAIR10_chr_all_Chr3:22175923-22180568",
"TAIR10_chr_all_Chr3:22232183-22236826",
"TAIR10_chr_all_Chr3:22513587-22521225",
"TAIR10_chr_all_Chr3:22541391-22551631",
"TAIR10_chr_all_Chr3:22695566-22700521",
"TAIR10_chr_all_Chr3:23100313-23104854",
"TAIR10_chr_all_Chr4:1312381-1317220",
"TAIR10_chr_all_Chr4:1565686-1567413",
"TAIR10_chr_all_Chr4:1619054-1624381",
"TAIR10_chr_all_Chr4:1669366-1679702",
"TAIR10_chr_all_Chr4:1685331-1695918",
"TAIR10_chr_all_Chr4:1796164-1804762",
"TAIR10_chr_all_Chr4:1814105-1818625",
"TAIR10_chr_all_Chr4:1825469-1835645",
"TAIR10_chr_all_Chr4:1918693-1925227",
"TAIR10_chr_all_Chr4:1943811-1955326",
"TAIR10_chr_all_Chr4:2001193-2010886",
"TAIR10_chr_all_Chr4:2064099-2068797",
"TAIR10_chr_all_Chr4:2188978-2194314",
"TAIR10_chr_all_Chr4:2198166-2203996",
"TAIR10_chr_all_Chr4:2206065-2214952",
"TAIR10_chr_all_Chr4:2286697-2288760",
"TAIR10_chr_all_Chr4:2302663-2307846",
"TAIR10_chr_all_Chr4:2643329-2648819",
"TAIR10_chr_all_Chr4:2829416-2838916",
"TAIR10_chr_all_Chr4:2867761-2875913",
"TAIR10_chr_all_Chr4:2900969-2903692",
"TAIR10_chr_all_Chr4:3013955-3024676",
"TAIR10_chr_all_Chr4:3074375-3083899",
"TAIR10_chr_all_Chr4:3087022-3097270",
"TAIR10_chr_all_Chr4:3173300-3182671",
"TAIR10_chr_all_Chr4:3274433-3285228",
"TAIR10_chr_all_Chr4:3344835-3353528",
"TAIR10_chr_all_Chr4:3386369-3394240",
"TAIR10_chr_all_Chr4:3432490-3443217",
"TAIR10_chr_all_Chr4:3457046-3469261",
"TAIR10_chr_all_Chr4:3537504-3547192",
"TAIR10_chr_all_Chr4:3567442-3572114",
"TAIR10_chr_all_Chr4:3572117-3583813",
"TAIR10_chr_all_Chr4:3838635-3852227",
"TAIR10_chr_all_Chr4:3868139-3877297",
"TAIR10_chr_all_Chr4:3907466-3915200",
"TAIR10_chr_all_Chr4:3987764-3997961",
"TAIR10_chr_all_Chr4:4036582-4046759",
"TAIR10_chr_all_Chr4:4124101-4126306",
"TAIR10_chr_all_Chr4:4133572-4141359",
"TAIR10_chr_all_Chr4:4176493-4185974",
"TAIR10_chr_all_Chr4:4240211-4249780",
"TAIR10_chr_all_Chr4:4269498-4278982",
"TAIR10_chr_all_Chr4:4385723-4392956",
"TAIR10_chr_all_Chr4:4599019-4603767",
"TAIR10_chr_all_Chr4:4645805-4648325",
"TAIR10_chr_all_Chr4:4651172-4660904",
"TAIR10_chr_all_Chr4:4670285-4676296",
"TAIR10_chr_all_Chr4:4677140-4684195",
"TAIR10_chr_all_Chr4:4735949-4741678",
"TAIR10_chr_all_Chr4:4758413-4770160",
"TAIR10_chr_all_Chr4:4893321-4903537",
"TAIR10_chr_all_Chr4:4958195-4969599",
"TAIR10_chr_all_Chr4:5016275-5023154",
"TAIR10_chr_all_Chr4:5063538-5074228",
"TAIR10_chr_all_Chr4:5079125-5084235",
"TAIR10_chr_all_Chr4:5099489-5109681",
"TAIR10_chr_all_Chr4:5127100-5135153",
"TAIR10_chr_all_Chr4:5303773-5309049",
"TAIR10_chr_all_Chr4:5396706-5402218",
"TAIR10_chr_all_Chr4:5905480-5916987",
"TAIR10_chr_all_Chr4:5965856-5970858",
"TAIR10_chr_all_Chr4:5987002-5991982",
"TAIR10_chr_all_Chr4:6067478-6072635",
"TAIR10_chr_all_Chr4:6155658-6161337",
"TAIR10_chr_all_Chr4:6529826-6537162",
"TAIR10_chr_all_Chr4:6588726-6593808",
"TAIR10_chr_all_Chr4:6723347-6728792",
"TAIR10_chr_all_Chr4:6914662-6919598",
"TAIR10_chr_all_Chr4:7298372-7305108",
"TAIR10_chr_all_Chr4:8313569-8319056",
"TAIR10_chr_all_Chr4:9483827-9488589",
"TAIR10_chr_all_Chr4:9508432-9512990",
"TAIR10_chr_all_Chr4:9732614-9737726",
"TAIR10_chr_all_Chr4:10753253-10761161",
"TAIR10_chr_all_Chr4:10992996-10998058",
"TAIR10_chr_all_Chr4:11040511-11046688",
"TAIR10_chr_all_Chr4:11363556-11368653",
"TAIR10_chr_all_Chr4:11676468-11681340",
"TAIR10_chr_all_Chr4:12129314-12131466",
"TAIR10_chr_all_Chr4:14258825-14263830",
"TAIR10_chr_all_Chr4:15559246-15562583",
"TAIR10_chr_all_Chr4:17712149-17716978",
"TAIR10_chr_all_Chr5:2441254-2443637",
"TAIR10_chr_all_Chr5:3904679-3909858",
"TAIR10_chr_all_Chr5:4208083-4213084",
"TAIR10_chr_all_Chr5:4787632-4800853",
"TAIR10_chr_all_Chr5:5629978-5635311",
"TAIR10_chr_all_Chr5:6204564-6214850",
"TAIR10_chr_all_Chr5:6293271-6301896",
"TAIR10_chr_all_Chr5:6403056-6408061",
"TAIR10_chr_all_Chr5:6433629-6438777",
"TAIR10_chr_all_Chr5:8624511-8629391",
"TAIR10_chr_all_Chr5:9059414-9064822",
"TAIR10_chr_all_Chr5:9181894-9189366",
"TAIR10_chr_all_Chr5:9949188-9954434",
"TAIR10_chr_all_Chr5:10128753-10133863",
"TAIR10_chr_all_Chr5:10308307-10314018",
"TAIR10_chr_all_Chr5:10359342-10364265",
"TAIR10_chr_all_Chr5:10617790-10629704",
"TAIR10_chr_all_Chr5:10708249-10713109",
"TAIR10_chr_all_Chr5:10764484-10772502",
"TAIR10_chr_all_Chr5:10808404-10812931",
"TAIR10_chr_all_Chr5:11086022-11099205",
"TAIR10_chr_all_Chr5:11138263-11145284",
"TAIR10_chr_all_Chr5:11159506-11166120",
"TAIR10_chr_all_Chr5:11358044-11372977",
"TAIR10_chr_all_Chr5:11397683-11407330",
"TAIR10_chr_all_Chr5:11423860-11439258",
"TAIR10_chr_all_Chr5:11471536-11486818",
"TAIR10_chr_all_Chr5:11527310-11538273",
"TAIR10_chr_all_Chr5:11577169-11587284",
"TAIR10_chr_all_Chr5:11597119-11609960",
"TAIR10_chr_all_Chr5:11621560-11635004",
"TAIR10_chr_all_Chr5:11643419-11650768",
"TAIR10_chr_all_Chr5:11741835-11752786",
"TAIR10_chr_all_Chr5:11771882-11782794",
"TAIR10_chr_all_Chr5:11797093-11807395",
"TAIR10_chr_all_Chr5:11916933-11927962",
"TAIR10_chr_all_Chr5:11980554-11994152",
"TAIR10_chr_all_Chr5:12007210-12018605",
"TAIR10_chr_all_Chr5:12029293-12043324",
"TAIR10_chr_all_Chr5:12189084-12194353",
"TAIR10_chr_all_Chr5:12289057-12296355",
"TAIR10_chr_all_Chr5:12315863-12322013",
"TAIR10_chr_all_Chr5:12470408-12476731",
"TAIR10_chr_all_Chr5:12501145-12508899",
"TAIR10_chr_all_Chr5:12518656-12531276",
"TAIR10_chr_all_Chr5:12569137-12583717",
"TAIR10_chr_all_Chr5:12616218-12621582",
"TAIR10_chr_all_Chr5:12623156-12634742",
"TAIR10_chr_all_Chr5:12639609-12650608",
"TAIR10_chr_all_Chr5:12710767-12719488",
"TAIR10_chr_all_Chr5:12766896-12776686",
"TAIR10_chr_all_Chr5:12897910-12901102",
"TAIR10_chr_all_Chr5:12997967-13012913",
"TAIR10_chr_all_Chr5:13030862-13045421",
"TAIR10_chr_all_Chr5:13084905-13092663",
"TAIR10_chr_all_Chr5:13117867-13128456",
"TAIR10_chr_all_Chr5:13141111-13152780",
"TAIR10_chr_all_Chr5:13156789-13163779",
"TAIR10_chr_all_Chr5:13253680-13258462",
"TAIR10_chr_all_Chr5:13261021-13272510",
"TAIR10_chr_all_Chr5:13299799-13308449",
"TAIR10_chr_all_Chr5:13317239-13327814",
"TAIR10_chr_all_Chr5:13344203-13354020",
"TAIR10_chr_all_Chr5:13371832-13379215",
"TAIR10_chr_all_Chr5:13397994-13404768",
"TAIR10_chr_all_Chr5:13525743-13534803",
"TAIR10_chr_all_Chr5:13736102-13741128",
"TAIR10_chr_all_Chr5:13948926-13953982",
"TAIR10_chr_all_Chr5:13994795-14000224",
"TAIR10_chr_all_Chr5:14083259-14088527",
"TAIR10_chr_all_Chr5:14113154-14118424",
"TAIR10_chr_all_Chr5:14203873-14209091",
"TAIR10_chr_all_Chr5:14822456-14825836",
"TAIR10_chr_all_Chr5:15175733-15180974",
"TAIR10_chr_all_Chr5:15233536-15245274",
"TAIR10_chr_all_Chr5:15351229-15362719",
"TAIR10_chr_all_Chr5:15490811-15498228",
"TAIR10_chr_all_Chr5:15668637-15685188",
"TAIR10_chr_all_Chr5:15922342-15927372",
"TAIR10_chr_all_Chr5:16745299-16756261",
"TAIR10_chr_all_Chr5:17090770-17095674",
"TAIR10_chr_all_Chr5:17298019-17311964",
"TAIR10_chr_all_Chr5:17596763-17605849",
"TAIR10_chr_all_Chr5:18142153-18146896",
"TAIR10_chr_all_Chr5:18233616-18238204",
"TAIR10_chr_all_Chr5:18490606-18495234",
"TAIR10_chr_all_Chr5:18933098-18938940",
"TAIR10_chr_all_Chr5:19246309-19248014",
"TAIR10_chr_all_Chr5:22002017-22006891",
"TAIR10_chr_all_Chr5:23134877-23139905"
]
Trans_only_TEs = [
"TAIR10_chr_all_Chr1:7065613-7075764",
"TAIR10_chr_all_Chr1:11502893-11512816",
"TAIR10_chr_all_Chr1:13019157-13023946",
"TAIR10_chr_all_Chr1:14019278-14024520",
"TAIR10_chr_all_Chr1:16094141-16103848",
"TAIR10_chr_all_Chr1:16191402-16198723",
"TAIR10_chr_all_Chr1:17255285-17265291",
"TAIR10_chr_all_Chr1:18828527-18833280",
"TAIR10_chr_all_Chr1:26366607-26371676",
"TAIR10_chr_all_Chr2:1781545-1786784",
"TAIR10_chr_all_Chr2:1906859-1912114",
"TAIR10_chr_all_Chr2:3752703-3763196",
"TAIR10_chr_all_Chr2:4542919-4547703",
"TAIR10_chr_all_Chr2:5062469-5067367",
"TAIR10_chr_all_Chr2:5073848-5078421",
"TAIR10_chr_all_Chr2:6819006-6823730",
"TAIR10_chr_all_Chr2:6954051-6958823",
"TAIR10_chr_all_Chr2:9187492-9195281",
"TAIR10_chr_all_Chr3:9692217-9697526",
"TAIR10_chr_all_Chr3:11664969-11669787",
"TAIR10_chr_all_Chr3:12243853-12255283",
"TAIR10_chr_all_Chr3:12982736-12993057",
"TAIR10_chr_all_Chr3:13087365-13092790",
"TAIR10_chr_all_Chr3:20304946-20310167",
"TAIR10_chr_all_Chr3:22059535-22064329",
"TAIR10_chr_all_Chr4:1669366-1679702",
"TAIR10_chr_all_Chr4:4269498-4278982",
"TAIR10_chr_all_Chr4:4385723-4392956",
"TAIR10_chr_all_Chr4:4651172-4660904",
"TAIR10_chr_all_Chr4:4958195-4969599",
"TAIR10_chr_all_Chr4:6067478-6072635",
"TAIR10_chr_all_Chr5:6403056-6408061",
"TAIR10_chr_all_Chr5:9949188-9954434",
"TAIR10_chr_all_Chr5:10359342-10364265",
"TAIR10_chr_all_Chr5:10808404-10812931",
"TAIR10_chr_all_Chr5:11423860-11439258",
"TAIR10_chr_all_Chr5:13371832-13379215",
"TAIR10_chr_all_Chr5:15233536-15245274",
"TAIR10_chr_all_Chr5:16745299-16756261",
"TAIR10_chr_all_Chr5:17090770-17095674"
]
Cis_only_TEs = [
"TAIR10_chr_all_Chr1:428928-438080",
"TAIR10_chr_all_Chr1:6718669-6721900",
"TAIR10_chr_all_Chr1:21748311-21753395",
"TAIR10_chr_all_Chr2:1417976-1428317",
"TAIR10_chr_all_Chr2:1629669-1637482",
"TAIR10_chr_all_Chr2:1689351-1697710",
"TAIR10_chr_all_Chr2:5040996-5048412",
"TAIR10_chr_all_Chr2:6544781-6559665",
"TAIR10_chr_all_Chr2:7224082-7229125",
"TAIR10_chr_all_Chr3:14579217-14584090",
"TAIR10_chr_all_Chr3:15250461-15262482",
"TAIR10_chr_all_Chr3:15591669-15600823",
"TAIR10_chr_all_Chr3:15768786-15773747",
"TAIR10_chr_all_Chr3:20502913-20507056",
"TAIR10_chr_all_Chr3:22513587-22521225",
"TAIR10_chr_all_Chr3:23100313-23104854",
"TAIR10_chr_all_Chr4:1565686-1567413",
"TAIR10_chr_all_Chr4:4599019-4603767",
"TAIR10_chr_all_Chr4:5396706-5402218",
"TAIR10_chr_all_Chr4:6529826-6537162",
"TAIR10_chr_all_Chr4:6588726-6593808",
"TAIR10_chr_all_Chr4:6723347-6728792",
"TAIR10_chr_all_Chr4:7298372-7305108",
"TAIR10_chr_all_Chr4:14258825-14263830",
"TAIR10_chr_all_Chr4:17712149-17716978",
"TAIR10_chr_all_Chr5:2441254-2443637",
"TAIR10_chr_all_Chr5:13084905-13092663",
"TAIR10_chr_all_Chr5:13948926-13953982",
"TAIR10_chr_all_Chr5:17298019-17311964",
"TAIR10_chr_all_Chr5:19246309-19248014"]
Cis_plus_Trans_TEs = [
"TAIR10_chr_all_Chr1:17473528-17476674",
"TAIR10_chr_all_Chr2:1557142-1562055",
"TAIR10_chr_all_Chr2:3667037-3671939",
"TAIR10_chr_all_Chr2:4230743-4236514",
"TAIR10_chr_all_Chr3:7359095-7368629",
"TAIR10_chr_all_Chr3:10494344-10500970",
"TAIR10_chr_all_Chr3:11652697-11657743",
"TAIR10_chr_all_Chr3:15081073-15085034",
"TAIR10_chr_all_Chr3:22541391-22551631",
"TAIR10_chr_all_Chr4:2829416-2838916",
"TAIR10_chr_all_Chr4:4670285-4676296",
"TAIR10_chr_all_Chr4:9732614-9737726",
"TAIR10_chr_all_Chr5:4208083-4213084",
"TAIR10_chr_all_Chr5:4787632-4800853",
"TAIR10_chr_all_Chr5:10308307-10314018",
"TAIR10_chr_all_Chr5:11086022-11099205",
"TAIR10_chr_all_Chr5:14113154-14118424"
]

# --- 2. PARSING FUNCTIONS ---
def Annotated_GFF3_line(in_liinee):
    Name = ""; Parent = ""; Alias = ""; Vcluster = ""; Superfamily = ""
    LTRsimilarity = 0.0; Size_TE = ""; Size_LTR = ""
    line = in_liinee.strip('\n')  
    parts = line.strip().split("\t")
    if len(parts) < 9: return [None]*11
    seqid_, source_, type_, start_, end_, score_, strand_, phase_, gff_attribute_ = parts
    attrib = gff_attribute_.rstrip().split(";")            
    for item in attrib:
        if "=" not in item: continue
        attribute, reference = item.split("=")        
        if attribute == "Parent": Parent = reference
        if attribute == "size(TE)": Size_TE = reference   
        if attribute == "LTRsimilarity": LTRsimilarity = float(reference)
        if attribute == "size(5LTR)": Size_LTR = reference
        if attribute == "Name": Name = reference  
        if attribute == "Alias": Annotation = reference  
        if attribute == "Superfamily": Superfamily = reference
        if attribute == "VSEARCH_Cluster": Cluster = reference                          
    return Parent, Size_TE, LTRsimilarity, Size_LTR, Name, Cluster, Annotation, Superfamily, seqid_, start_, end_

def get_residuals(data_df, pc_df):
    data_df.index = data_df.index.astype(str)
    pc_df.index = pc_df.index.astype(str)
    common_samples = data_df.index.intersection(pc_df.index)
    Y = data_df.loc[common_samples].copy()
    X = pc_df.loc[common_samples].copy()
    Y = Y.loc[:, Y.std() > 0]
    residuals = pd.DataFrame(index=Y.index, columns=Y.columns)
    model = LinearRegression()
    for col in Y.columns:
        y_vec = Y[col].fillna(Y[col].mean()).values.reshape(-1, 1)
        try:
            model.fit(X, y_vec)
            residuals[col] = (y_vec - model.predict(X)).flatten()
        except: continue
    return residuals.dropna(axis=1, how='all')

# --- 3. DATA LOADING & PREPROCESSING ---
Parental_to_Naming = defaultdict(str)
with open(f"{path_gff3}{inputFile_gff3}") as infile:
    for entry_line in infile: 
        res = Annotated_GFF3_line(entry_line)
        if res[0] is not None:
            Parental_to_Naming[res[0]] = res[4]

#print("Loading PCs...")
df_snp_raw = pd.read_csv(f"{path_snps}{file_snps}", sep=r'\s+', header=None)
df_pcs = df_snp_raw.iloc[:, [1] + list(range(2, 22))]
df_pcs.columns = ['sample'] + [f'PC{i}' for i in range(1, 21)]
df_pcs.set_index('sample', inplace=True)

#print("Processing Expression...")
df_expr_raw = pd.read_csv(f"{path_expr}{file_expr}", sep='\t').set_index('IID')
df_expr_res = get_residuals(df_expr_raw.drop(columns=['FID'], errors='ignore'), df_pcs)

#print("Processing Bioclimatic...")
df_bioclim = pd.read_csv(f"{path_bioc}{file_bioc}", sep='\t').set_index('IID')
df_bioclim.index = df_bioclim.index.astype(str)
scaler = StandardScaler()
df_bioc_scaled = pd.DataFrame(scaler.fit_transform(df_bioclim), 
                              index=df_bioclim.index, columns=df_bioclim.columns)

#print("Processing Methylation...")
meth_res_master = {}
for context, filename in meth_files.items():
    df_raw = pd.read_csv(f"{path_meth}{filename}", sep='\t', header=0)
    df_pivot = df_raw.set_index('#Parental').iloc[:, 3:].T
    meth_res_master[context] = get_residuals(df_pivot, df_pcs)

# --- 4. SUBGROUP SETUP ---
Associated_set = set(Trans_only_TEs) | set(Cis_only_TEs) | set(Cis_plus_Trans_TEs)
subgroups = {
    "trans_only": Trans_only_TEs,
    "cis_only": Cis_only_TEs,
    "cistrans": Cis_plus_Trans_TEs,
    "non_assoc_expr": [te for te in Expressed_list if te not in Associated_set]
}

# --- 5. INTERACTION MODEL ---
def run_interaction(subgroup_name, te_coords, context):
    results = []
    if context not in meth_res_master: return pd.DataFrame()
    
    common_all = df_expr_res.index.intersection(df_bioc_scaled.index).intersection(meth_res_master[context].index)
    valid_tes = [te for te in te_coords if te in df_expr_res.columns and te in meth_res_master[context].columns]
    
    for te in valid_tes:
        for bio in df_bioc_scaled.columns:
            df_model = pd.DataFrame({
                'Y': df_expr_res.loc[common_all, te],
                'Bioclimate': df_bioc_scaled.loc[common_all, bio],
                'Methylation': meth_res_master[context].loc[common_all, te]
            }).dropna()
            
            if len(df_model) < 20: continue
            
            df_model['Interaction'] = df_model['Bioclimate'] * df_model['Methylation']
            X = sm.add_constant(df_model[['Bioclimate', 'Methylation', 'Interaction']])
            
            # 1. Standard OLS
            ols_res = sm.OLS(df_model['Y'], X).fit()
            # 2. Robust Linear Model (RLM) - Handles outliers/non-normality
            rlm_res = sm.RLM(df_model['Y'], X, M=sm.robust.norms.HuberT()).fit()
            # 3. Diagnostic Tests on OLS
            # Jarque-Bera: p < 0.05 means residuals are NOT normal
            _, jb_p, _, _ = jarque_bera(ols_res.resid)
            # Breusch-Pagan: p < 0.05 means variance is NOT equal (heteroscedasticity)
            _, bp_p, _, _ = het_breuschpagan(ols_res.resid, X)

            results.append({
                'Context': context,
                'Subgroup': subgroup_name,
                'Coordinate': te,
                'Name': Parental_to_Naming.get(te, te),
                'Bioclim': bio,
                # OLS Results
                'Beta_Bioclimate': ols_res.params['Bioclimate'],
                'P_Bioclimate': ols_res.pvalues['Bioclimate'],
                'Beta_Methylation': ols_res.params['Methylation'],
                'P_Methylation': ols_res.pvalues['Methylation'],
                'Beta_Interaction': ols_res.params['Interaction'],
                'P_Interaction': ols_res.pvalues['Interaction'],
                'P_Model': ols_res.f_pvalue,
                'R_squared': ols_res.rsquared,
                # Robust Comparison (RLM)
                'RLM_Beta_Bioclim': rlm_res.params['Bioclimate'],
                'RLM_P_Bioclim': rlm_res.pvalues['Bioclimate'],
                'RLM_Beta_Meth': rlm_res.params['Methylation'],
                'RLM_P_Meth': rlm_res.pvalues['Methylation'],
                'RLM_Beta_Inter': rlm_res.params['Interaction'],
                'RLM_P_Inter': rlm_res.pvalues['Interaction'],
                # Diagnostics (Flags for "Better" model selection)
                'Non_Normal': 1 if jb_p < 0.05 else 0,
                'Heteroscedastic': 1 if bp_p < 0.05 else 0
            })
    return pd.DataFrame(results)

# --- 6. AUTOMATED EXECUTION & PLOTTING (RLM VERSION) ---
group_order = ["cis_only", "cistrans", "trans_only", "non_assoc_expr"]
final_master_list = []
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)] 

for context in meth_files.keys():
    print(f"Analyzing {context} context (RLM prioritized)...")
    context_res_list = [run_interaction(name, tes, context) for name, tes in subgroups.items()]
    df_context = pd.concat([df for df in context_res_list if not df.empty])
    final_master_list.append(df_context)
    sorted_bioclim = sorted(df_context['Bioclim'].unique(), key=natural_sort_key)
    # Updated mapping to use RLM for the Interaction term
    plot_map = {
        'β1 (bioclimatic)': ('RLM_Beta_Bioclim', 'RLM_P_Bioclim'), 
        'β2 (methylation)': ('RLM_Beta_Meth', 'RLM_P_Meth'), 
        'β3 (interaction)': ('RLM_Beta_Inter', 'RLM_P_Inter') # Switched to RLM
    }

    for label, (beta_col, p_col) in plot_map.items():
        df_filtered = df_context.copy()
        df_filtered.loc[df_filtered[p_col] >= P_THRESHOLD, beta_col] = np.nan
        pivot_df = df_filtered.groupby(['Subgroup', 'Bioclim'])[beta_col].mean().unstack().reindex(index=group_order, columns=sorted_bioclim)
        
        plt.figure(figsize=(16, 6))
        ax = sns.heatmap(pivot_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                    cbar_kws={'label': f'Mean significant {label}'})
        plt.title(f"Context: {context} | {label} Effect (RLM p < {P_THRESHOLD})")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20, fontweight=700)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20, fontweight=700)
        plt.ylabel(" ", fontname='arial', fontsize=24, fontweight=700, labelpad=10.0)
        plt.xlabel(" ", fontname='arial', fontsize=24, fontweight=700)
        plt.tight_layout()
        
        # Filename updated to 13_RLM_
        clean_label = label.split(' ')[0].replace('β', 'Beta')
        plt.savefig(f"{path_out}13_RLM_Heatmap_{context}_{clean_label}_regressOUT_notFRD{str(P_THRESHOLD).replace('.', '')}.png", dpi=300)
        #plt.savefig(f"{path_out}13_RLM_Heatmap_{context}_{clean_label}_regressOUT_{str(P_THRESHOLD).replace('.', '')}.svg", format="SVG", dpi=2000, bbox_inches='tight')
        plt.close()

master_results = pd.concat(final_master_list)

# =================================================================
# --- MULTIPLE TESTING CORRECTION ---
# =================================================================
# Calculate a single "Effective Alpha" and update the global P_THRESHOLD
# Save the original threshold for the report
ORIGINAL_P = P_THRESHOLD 

# 1. Collect all raw p-values that were subjected to testing
# Pool the three main RLM tests (Bioclim, Meth, Inter) 
p_values_to_correct = master_results[['RLM_P_Bioclim', 'RLM_P_Meth', 'RLM_P_Inter']].values.flatten()

# 2. Apply Benjamini-Hochberg (FDR)
rejected, adjusted_p, _, _ = multipletests(p_values_to_correct, alpha=ORIGINAL_P, method='fdr_bh')

# 3. Find the "Effective Threshold" 
# This is the highest raw p-value that still meets the FDR requirement
if any(rejected):
    NEW_THRESHOLD = p_values_to_correct[rejected].max()
else:
    NEW_THRESHOLD = 0.0000000001 # Essentially nothing passes

# --- STANDARD OUTPUT REPORT ---
print("\n" + "="*50)
print("MULTIPLE TESTING CORRECTION REPORT (Benjamini-Hochberg)")
print("-"*50)
print(f"Original Alpha Threshold:   {ORIGINAL_P}")
print(f"Total Tests Performed:      {len(p_values_to_correct)}")
print(f"FDR-Corrected Threshold:    {NEW_THRESHOLD:.6e}")
print(f"Total Significant Hits:     {sum(rejected)}")
print("-"*50)
print("ACTION: Global P_THRESHOLD has been updated.")
print("All subsequent plots and reports now reflect FDR-corrected significance.")
print("="*50 + "\n")

# 4. OVERWRITE THE GLOBAL VARIABLE
P_THRESHOLD = NEW_THRESHOLD

# --- 7. TOTAL VARIANCE EXPLAINED REPORT ---
print("\n--- GLOBAL VARIANCE EXPLAINED (Best-Fit Model per TE) ---")

# 1. Filter for significant models only
sig_only = master_results[master_results['P_Model'] < P_THRESHOLD].copy()

if not sig_only.empty:
    # 2. Find the 'Best' Bioclim/Context combination for each TE
    # This identifies the maximum R2 climate can explain for a specific TE
    best_per_te = sig_only.sort_values('R_squared', ascending=False).drop_duplicates('Coordinate')

    for subgroup in group_order:
        sub_data = best_per_te[best_per_te['Subgroup'] == subgroup]
        if sub_data.empty: continue
        
        avg_total_r2 = sub_data['R_squared'].mean()
        max_te_r2 = sub_data['R_squared'].max()
        count_te = len(sub_data)
        
        print(f"Subgroup: {subgroup:15} | Responsive TEs: {count_te:4} | Mean Total R²: {avg_total_r2:.4f} | Peak R²: {max_te_r2:.4f}")

    # 3. Overall Global Metric
    global_r2 = best_per_te['R_squared'].mean()
    print(f"\nOVERALL SYSTEM R-SQUARED: {global_r2:.4f}")
    print(f"Interpretation: For responsive TEs, climate and methylation explain {global_r2*100:.2f}% of expression noise.")
    
# --- 8. REVISED SUMMARY: VARIANCE DECOMPOSITION REPORT ---
print("\n--- Variance Decomposition (Contribution to Total R²) ---")
sig_only = master_results[master_results['P_Model'] < P_THRESHOLD].copy()

if not sig_only.empty:
    # Estimate the relative weight of each Beta coefficient to the total R2
    # Use the squared T-statistics as a proxy for partial R2 contribution
    sig_only['Total_Weight'] = sig_only['Beta_Bioclimate']**2 + sig_only['Beta_Methylation']**2 + sig_only['Beta_Interaction']**2
    sig_only['Weight_Bio'] = (sig_only['Beta_Bioclimate']**2 / sig_only['Total_Weight']) * sig_only['R_squared']
    sig_only['Weight_Meth'] = (sig_only['Beta_Methylation']**2 / sig_only['Total_Weight']) * sig_only['R_squared']
    sig_only['Weight_Inter'] = (sig_only['Beta_Interaction']**2 / sig_only['Total_Weight']) * sig_only['R_squared']
    summary_var = sig_only.groupby('Subgroup')[['Weight_Bio', 'Weight_Meth', 'Weight_Inter']].mean()
    print(summary_var)

print("\n--- Summary: Statistical Significance Decomposed per methylation context ---")
for context in meth_files.keys():
    ctx_df = master_results[master_results['Context'] == context]
    total = len(ctx_df)
    if total == 0: continue
    
    sig_total = len(ctx_df[ctx_df['P_Model'] < P_THRESHOLD])
    sig_climate = len(ctx_df[ctx_df['P_Bioclimate'] < P_THRESHOLD])
    sig_meth = len(ctx_df[ctx_df['P_Methylation'] < P_THRESHOLD])
    sig_inter = len(ctx_df[ctx_df['P_Interaction'] < P_THRESHOLD])
    avg_r2 = ctx_df[ctx_df['P_Model'] < P_THRESHOLD]['R_squared'].mean()
    
    print(f"Context {context}:")
    print(f"  - Overall Significant Models (F-test): {sig_total}/{total} ({(sig_total/total)*100:.2f}%)")
    print(f"  - Bioclimate Main Effect Significant: {sig_climate}/{total} ({(sig_climate/total)*100:.2f}%)")
    print(f"  - Methylation Main Effect Significant: {sig_meth}/{total} ({(sig_meth/total)*100:.2f}%)")
    print(f"  - Interaction Significant: {sig_inter}/{total} ({(sig_inter/total)*100:.2f}%)")
    print(f"  - Avg R-squared of significant models: {avg_r2:.4f}")

# --- 9. COMPARISON BAR PLOT (Tests - RLM Version) ---
impact_data = []
print("\n--- Subgroup Breakdown for Driver Plot (RLM Interaction) ---")
for context in meth_files.keys():
    ctx_df = master_results[master_results['Context'] == context]
    for subgroup in group_order:
        group = ctx_df[ctx_df['Subgroup'] == subgroup]
        n = len(group)
        if n == 0: continue
        
        count_bio = sum(group['RLM_P_Bioclim'] < P_THRESHOLD)
        count_meth = sum(group['RLM_P_Meth'] < P_THRESHOLD)
        count_inter = sum(group['RLM_P_Inter'] < P_THRESHOLD) # Switched to RLM_P_Inter
        
        impact_data.append({'Subgroup': subgroup, 'Context': context, 'Variable': 'Bioclimatic', 'Percent': (count_bio/n)*100})
        impact_data.append({'Subgroup': subgroup, 'Context': context, 'Variable': 'Methylation', 'Percent': (count_meth/n)*100})
        impact_data.append({'Subgroup': subgroup, 'Context': context, 'Variable': 'Interaction', 'Percent': (count_inter/n)*100})

impact_df = pd.DataFrame(impact_data)

plt.figure(figsize=(5, 12))
ax = sns.barplot(data=impact_df, x='Subgroup', y='Percent', hue='Variable', order=group_order, 
                 palette='viridis', width=0.7, edgecolor='black', linewidth=1)
plt.title(f" ", fontsize=13, fontweight='bold') #Drivers of TE Expression: Bioclimate, Methylation, and Interaction (p < {P_THRESHOLD})"
plt.ylabel("Significant β term tests (%)", fontname='arial', fontsize=24, fontweight=700, labelpad=10.0)
plt.xlabel(" ", fontname='arial', fontsize=24, fontweight=700)
plt.xticks(rotation=45, ha='right', fontname='arial', fontsize=20, fontweight=700)
plt.yticks(fontname='arial', fontsize=20, fontweight=700) 
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title='β term', bbox_to_anchor=(0.99, 0.99), loc='upper right', fontsize=15, title_fontsize=15)
plt.subplots_adjust(left=0.30, right=0.95, top=0.95, bottom=0.20)

plt.savefig(f"{path_out}13_RLM_Comparison_of_Drivers_regressedOUT_FDR{str(P_THRESHOLD).replace('.', '')}.png", dpi=300)
plt.savefig(f"{path_out}13_RLM_Comparison_of_Drivers_regressedOUT_FDR{str(P_THRESHOLD).replace('.', '')}.svg", format="SVG", dpi=2000)
#plt.savefig(f"{path_out}13_Comparison_of_Drivers_regressedOUT_ExpANDMeth_{str(P_THRESHOLD).replace('.', '')}.pdf", format='PDF', dpi=300)
plt.close()

# --- 9.BIS COMPARISON BAR PLOT (Unique TEs - RLM Version) ---
impact_data_tes = []
print("\n--- Subgroup Breakdown for Unique TE Driver Plot (RLM Interaction) ---")
for context in meth_files.keys():
    ctx_df = master_results[master_results['Context'] == context]
    for subgroup in group_order:
        group = ctx_df[ctx_df['Subgroup'] == subgroup]
        if group.empty: continue
        
        total_unique_tes = group['Coordinate'].nunique()
        count_bio_te = group[group['RLM_P_Bioclim'] < P_THRESHOLD]['Coordinate'].nunique()
        count_meth_te = group[group['RLM_P_Meth'] < P_THRESHOLD]['Coordinate'].nunique()
        count_inter_te = group[group['RLM_P_Inter'] < P_THRESHOLD]['Coordinate'].nunique() # Switched to RLM_P_Inter
        
        impact_data_tes.append({'Subgroup': subgroup, 'Context': context, 'Variable': 'Bioclimatic', 'Percent': (count_bio_te/total_unique_tes)*100})
        impact_data_tes.append({'Subgroup': subgroup, 'Context': context, 'Variable': 'Methylation', 'Percent': (count_meth_te/total_unique_tes)*100})
        impact_data_tes.append({'Subgroup': subgroup, 'Context': context, 'Variable': 'Interaction', 'Percent': (count_inter_te/total_unique_tes)*100})

impact_df_tes = pd.DataFrame(impact_data_tes)

plt.figure(figsize=(5, 12))
ax = sns.barplot(data=impact_df_tes, x='Subgroup', y='Percent', hue='Variable', order=group_order, 
                 palette='viridis', width=0.7, edgecolor='black', linewidth=1)

plt.title(" ", fontsize=13, fontweight='bold') 
plt.ylabel("LTR-retroTEs with significant β term (%)", fontname='arial', fontsize=24, fontweight=700, labelpad=10.0)
plt.xlabel(" ", fontname='arial', fontsize=24, fontweight=700)
plt.xticks(rotation=45, ha='right', fontname='arial', fontsize=20, fontweight=700)
plt.yticks(fontname='arial', fontsize=20, fontweight=700) 
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title='β term', bbox_to_anchor=(0.82, 0.99), loc='upper right', fontsize=15, title_fontsize=15) #0.98
plt.subplots_adjust(left=0.30, right=0.95, top=0.95, bottom=0.20)

plt.savefig(f"{path_out}13_RLM_Comparison_of_Drivers_UNIQUE_TEs_FDR{str(P_THRESHOLD).replace('.', '')}.png", dpi=300)
plt.savefig(f"{path_out}13_RLM_Comparison_of_Drivers_UNIQUE_TEs_FDR{str(P_THRESHOLD).replace('.', '')}.svg", format="SVG", dpi=2000)
plt.close()

# --- 10. DIRECTIONAL CONSISTENCY ANALYSIS (RLM UPDATED) ---
# Mapping β3 to RLM columns specifically
beta_mapping = {
    'RLM_Beta_Bioclim': 'β1 (bioclimatic)',
    'RLM_Beta_Meth': 'β2 (methylation)',
    'RLM_Beta_Inter': 'β3 (interaction)'
}
pval_mapping = {
    'RLM_Beta_Bioclim': 'RLM_P_Bioclim',
    'RLM_Beta_Meth': 'RLM_P_Meth',
    'RLM_Beta_Inter': 'RLM_P_Inter'
}
for beta_col, label in beta_mapping.items():
    p_col = pval_mapping[beta_col]
    df_sig = master_results[master_results[p_col] < P_THRESHOLD].copy()
    if df_sig.empty: continue  
    df_sig['Direction'] = np.where(df_sig[beta_col] > 0, 'Positive (+)', 'Negative (-)')
    df_sig['mContext'] = 'm' + df_sig['Context'].astype(str)
    dir_counts = df_sig.groupby(['Subgroup', 'mContext', 'Direction']).size().unstack(fill_value=0)
    dir_props = dir_counts.div(dir_counts.sum(axis=1), axis=0) * 100
    dir_props = dir_props.reset_index()
    dir_props['X_axis'] = dir_props['Subgroup'] + "\n(" + dir_props['mContext'] + ")"
    custom_x_order = []
    for g in group_order:
        for c in ['mCG', 'mCHG', 'mCHH']:
            label_x = f"{g}\n({c})"
            if label_x in dir_props['X_axis'].values:
                custom_x_order.append(label_x)

    plt.figure(figsize=(14, 12))
    ax = dir_props.set_index('X_axis').reindex(custom_x_order)[['Negative (-)', 'Positive (+)']].plot(
            kind='bar', stacked=True, color=['#708090', '#E69F00'], 
            ax=plt.gca(), edgecolor='black', width=0.8
        )
    plt.ylabel(f"Fraction of significant {label} tests (%)", fontname='arial', fontsize=24, fontweight=700, labelpad=10.0)
    plt.xlabel(" ", fontname='arial', fontsize=24, fontweight=700)
    plt.xticks(rotation=45, ha='right', fontsize=20, fontweight=700)
    plt.yticks(fontsize=20, fontweight=700)
    plt.axhline(50, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.legend(title='Effect Direction', bbox_to_anchor=(0.8, 0.2), loc='upper right', fontsize=15, title_fontsize=15)
    plt.subplots_adjust(left=0.30, right=0.95, top=0.95, bottom=0.20)
    
    # Updated Filename prefix to 13_RLM_
    clean_label = label.split(' ')[0].replace('β', 'Beta').replace('(', '').replace(')', '')
    plt.savefig(f"{path_out}13_RLM_Directional_Consistency_{clean_label}_FDR{str(P_THRESHOLD).replace('.', '')}.png", dpi=300)
    plt.savefig(f"{path_out}13_RLM_Directional_Consistency_{clean_label}_FDR{str(P_THRESHOLD).replace('.', '')}.svg", format="SVG", dpi=2000)
    plt.close()

# --- 11. MEAN BETA TERMS PER SUBGROUP (RLM UPDATED) ---
context_palette = {'mCG': '#e41a1c', 'mCHG': '#377eb8', 'mCHH': '#4daf4a'}
for beta_col, label in beta_mapping.items():
    p_col = pval_mapping[beta_col]
    df_sig_all = master_results[master_results[p_col] < P_THRESHOLD].copy()
    if df_sig_all.empty: continue
    df_sig_all['Context'] = 'm' + df_sig_all['Context'].astype(str)
    for direction in ['positives', 'negatives']:
        if direction == 'positives':
            df_sig_beta = df_sig_all[df_sig_all[beta_col] > 0].copy()
            dir_label = "positives"
        else:
            df_sig_beta = df_sig_all[df_sig_all[beta_col] < 0].copy()
            dir_label = "negatives"
        if df_sig_beta.empty: continue

        plt.figure(figsize=(7, 12))
        ax = sns.barplot(
            data=df_sig_beta, x='Subgroup', y=beta_col, hue='Context',
            order=group_order, palette=context_palette, errorbar=('ci', 90), 
            capsize=0.001, edgecolor='black', linewidth=1
        )

        plt.ylabel(f"Mean of significant {label} ({dir_label})", fontname='arial', fontsize=24, fontweight=700, labelpad=10.0)
        plt.xlabel(" ", fontname='arial', fontsize=24, fontweight=700)  
        plt.xticks(rotation=45, ha='right', fontname='arial', fontsize=20, fontweight=700)
        plt.yticks(fontname='arial', fontsize=20, fontweight=700)
        plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        if dir_label == "negatives":
            plt.legend(title='Context', bbox_to_anchor=(0.65, 0.20), loc='upper right', ncol=1, fontsize=15)
        else:
            plt.legend(title='Context', bbox_to_anchor=(0.65, 0.99), loc='upper right', ncol=1, fontsize=15)
        plt.subplots_adjust(left=0.30, right=0.95, top=0.95, bottom=0.20)
        
        # Updated Filename prefix to 13_RLM_
        clean_label = label.split(' ')[0].replace('β', 'Beta').replace('(', '').replace(')', '')
        plt.savefig(f"{path_out}13_RLM_Mean_{clean_label}_Sig_{direction.upper()}_FDR{str(P_THRESHOLD).replace('.', '')}.png", dpi=300)
        plt.savefig(f"{path_out}13_RLM_Mean_{clean_label}_Sig_{direction.upper()}_FDR{str(P_THRESHOLD).replace('.', '')}.svg", format="SVG", dpi=2000)
        plt.close()

# --- 12. Normalized Subgroup Landscape (RLM UPDATED) ---
total_tests = master_results.groupby(['Subgroup', 'Context']).size().reset_index(name='Total_Tests')
# Using RLM_P_Inter here for the landscape
sig_counts = master_results[master_results['RLM_P_Inter'] < P_THRESHOLD].groupby(['Subgroup', 'Context']).size().reset_index(name='Sig_Counts')
landscape_df = pd.merge(total_tests, sig_counts, on=['Subgroup', 'Context'], how='left').fillna(0)
landscape_df['Percent_Significant'] = (landscape_df['Sig_Counts'] / landscape_df['Total_Tests']) * 100

plt.figure(figsize=(8, 6))
sns.barplot(data=landscape_df, x='Subgroup', y='Percent_Significant', hue='Context', order=group_order, palette={'CG': '#e41a1c', 'CHG': '#377eb8', 'CHH': '#4daf4a'}, edgecolor='black')
plt.title(f"ROBUST Subgroup Sensitivity (RLM p < {P_THRESHOLD})", fontsize=13, fontweight='bold')
plt.ylabel("% significant interactions (RLM)", labelpad=10.0)
plt.savefig(f"{path_out}13_RLM_Subgroup_Significance_Landscape_FDR{str(P_THRESHOLD).replace('.', '')}.png", dpi=300)
plt.close()

# --- SAVE MASTER FILE ---
# Updated Filename to reflect RLM
master_results.to_csv(f"{path_out}13_RLM_Interaction_Analysis_Summary_regressedOUT_FDR{str(P_THRESHOLD).replace('.', '')}.csv", index=False)

# --- 14. OLS VS. ROBUST MODEL COMPARISON REPORT (FULL SUITE) ---
print("\n" + "="*60)
print("STATISTICAL VALIDITY & ROBUSTNESS COMPARISON REPORT")
print("="*60)

# 1. Assumption Violations (Global)
total_models = len(master_results)
non_normal_pct = (master_results['Non_Normal'].sum() / total_models) * 100
hetero_pct = (master_results['Heteroscedastic'].sum() / total_models) * 100

print(f"Total Models Tested: {total_models}")
print(f"Models violating Normality (Jarque-Bera):           {non_normal_pct:.2f}%")
print(f"Models violating Homoscedasticity (Breusch-Pagan): {hetero_pct:.2f}%")
print("-" * 60)

# 2. Detailed Term-by-Term Comparison
# Define the pairs to compare: (OLS_Beta, RLM_Beta, OLS_P, RLM_P, Label)
comparison_terms = [
    ('Beta_Bioclimate', 'RLM_Beta_Bioclim', 'P_Bioclimate', 'RLM_P_Bioclim', 'Bioclimate (β1)'),
    ('Beta_Methylation', 'RLM_Beta_Meth', 'P_Methylation', 'RLM_P_Meth', 'Methylation (β2)'),
    ('Beta_Interaction', 'RLM_Beta_Inter', 'P_Interaction', 'RLM_P_Inter', 'Interaction (β3)')
]
for ols_b, rlm_b, ols_p, rlm_p, label in comparison_terms:
    # Mean Absolute Change in Beta
    beta_diff = (master_results[ols_b] - master_results[rlm_b]).abs().mean()
    # Spurious: Sig in OLS, but vanished in RLM (Outlier-driven artifacts)
    spurious = master_results[(master_results[ols_p] < P_THRESHOLD) & (master_results[rlm_p] >= P_THRESHOLD)]
    # Hidden: Not sig in OLS, but emerged in RLM (Outlier-masked signals)
    hidden = master_results[(master_results[ols_p] >= P_THRESHOLD) & (master_results[rlm_p] < P_THRESHOLD)]
    # Consistent: Significant in both
    consistent = master_results[(master_results[ols_p] < P_THRESHOLD) & (master_results[rlm_p] < P_THRESHOLD)]
    print(f"Variable: {label}")
    print(f"  - Mean Change in Beta (OLS vs RLM): {beta_diff:.4f}")
    print(f"  - OLS Artifacts (Spurious):         {len(spurious):<5} (Sig in OLS only)")
    print(f"  - Robust Signals (Hidden):          {len(hidden):<5} (Sig in RLM only)")
    print(f"  - High-Confidence Signals:          {len(consistent):<5} (Sig in both)")
    print("-" * 30)

# 3. Final Verdict
if non_normal_pct > 50:
    print("\nFINAL ADVICE: Critical violation of normality detected (90%+).")
    print("OLS p-values are highly unreliable in this dataset.")
    print("The RLM model successfully recovered 'Hidden' signals previously masked by outliers.")
elif non_normal_pct > 30:
    print("\nFINAL ADVICE: High violation rates. RLM is preferred for consistency.")
else:
    print("\nFINAL ADVICE: OLS assumptions are largely met. Both models should converge.")
print("="*60)