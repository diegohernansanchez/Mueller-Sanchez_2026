"""
Runs in PYTHON 3 for Windows
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mofapy2.run.entry_point import entry_point
import h5py
import mofax as mfx
import seaborn as sns
import math

### --- 1. CONFIGURATION ---
MOFA_FACTORS = 9

path_ecotypes = "./PATH/"
file_ecotypes = "List_1135_Aras_ecotypes.txt"
path_gff3     = "./PATH/"
inputFile_gff3= "Allnoncanonical_plus_unique_FINAL_LTRharvest85_TAIR10_chr_all_NAMEnCLUSTER.gff3"

path_expr     = "./PATH/"
path_snps     = "./PATH/"
path_meth     = "./PATH/"
path_hdf5     = "./PATH/"

file_expr     = "6_TEs_expresion_model.txt"
file_snps     = "plink.eigenvec_ALL_ACCESSIONS_allsnps"
file_eigenval = "plink.eigenval_ALL_ACCESSIONS_allsnps" 
meth_files    = { # Dictionary to handle the three contexts
    'CG': 'MeC_bismark_LTRs_vs_accessions_CG.txt',
    'CHG': 'MeC_bismark_LTRs_vs_accessions_CHG.txt',
    'CHH': 'MeC_bismark_LTRs_vs_accessions_CHH.txt'}
hdf5_file     = "MOFA_Result_All_Contexts_RESIDUALS.hdf5"

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

# MASTER LISTS & GROUP INFERENCE ---
Trans_only_set = set([c.strip() for c in Trans_only_TEs])
Cis_only_set = set([c.strip() for c in Cis_only_TEs])
Cis_Trans_set = set([c.strip() for c in Cis_plus_Trans_TEs])
Expressed_set = set([c.strip() for c in Expressed_list])
# Combined "Associated" set
Associated_set = Trans_only_set | Cis_only_set | Cis_Trans_set

### --- 2. DATA LOADING & RESIDUAL FUNCTION ---
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

# Build the master renaming dictionary
Parental_to_Naming = defaultdict(str)
with open(f"{path_gff3}{inputFile_gff3}") as infile:
    for entry_line in infile: 
        res = Annotated_GFF3_line(entry_line)
        if res[0] is not None:
            Parental, size_TE, LTRsim, size_5LTR, Naming, Vcluster, Alias, Family, chromosome, start, end = res
            Parental_to_Naming[Parental] = Naming
print(f"GFF3 Parsed: Built Naming map for {len(Parental_to_Naming)} LTR-retrotransposons.")

def get_residuals(data_df, pc_df):
    data_df.index = data_df.index.astype(str)
    pc_df.index = pc_df.index.astype(str)
    common_samples = data_df.index.intersection(pc_df.index)
    Y = data_df.loc[common_samples].copy()
    X = pc_df.loc[common_samples].copy()
    Y = Y.dropna(axis=1, how='all')
    Y = Y.loc[:, Y.std() > 0]
    
    # Renaming step
    Y.columns = [c.strip() for c in Y.columns]
    # Keep the original coordinates as a temporary mapping attribute before renaming
    original_coords = Y.columns.tolist()
    Y = Y.rename(columns=Parental_to_Naming)
    
    residuals = pd.DataFrame(index=Y.index, columns=Y.columns)
    model = LinearRegression()
    for col in Y.columns:
        y_vec = Y[col].fillna(Y[col].mean()).values.reshape(-1, 1)
        try:
            model.fit(X, y_vec)
            residuals[col] = (y_vec - model.predict(X)).flatten()
        except:
            continue
    # Store original coordinates in the dataframe for splitting later
    residuals.columns.name = 'feature_name'
    return residuals.dropna(axis=1, how='any'), original_coords

# --- LOAD SNP PCs ---
df_snp_raw = pd.read_csv(f"{path_snps}{file_snps}", sep=r'\s+', header=None)
df_pcs = df_snp_raw.iloc[:, [1] + list(range(2, 22))]
df_pcs.columns = ['sample'] + [f'PC{i}' for i in range(1, 21)]
df_pcs.set_index('sample', inplace=True)

# --- LOAD & PROCESS EXPRESSION ---
print("Processing Expression residuals...")
df_expr_raw = pd.read_csv(f"{path_expr}{file_expr}", sep='\t').set_index('IID')
df_expr_res, expr_coords = get_residuals(df_expr_raw.drop(columns=['FID'], errors='ignore'), df_pcs)

# --- LOAD & PROCESS METHYLATION ---
meth_residuals_master = {}
meth_coords_master = {}
for context, filename in meth_files.items():
    print(f"Processing {context} residuals...")
    df_raw = pd.read_csv(f"{path_meth}{filename}", sep='\t', header=0)
    df_pivot = df_raw.set_index('#Parental').iloc[:, 3:].T
    res, coords = get_residuals(df_pivot, df_pcs)
    meth_residuals_master[context] = res
    meth_coords_master[context] = coords

### --- 3. SPLIT INTO CATEGORICAL VIEWS & REFORMAT ---
def split_and_format(res_df, coords_list, base_name):
    """Splits residuals into 5 categorical views and reports counts."""
    # Map coordinates to renamed columns
    coord_to_name = dict(zip(coords_list, res_df.columns))
    
    # Identify which names belong to which category
    cat_map = {
        'cis_only':     [coord_to_name[c] for c in coords_list if c in Cis_only_set],
        'cistrans':     [coord_to_name[c] for c in coords_list if c in Cis_Trans_set],
        'trans_only':   [coord_to_name[c] for c in coords_list if c in Trans_only_set],
        'non_assoc_expr': [coord_to_name[c] for c in coords_list if c not in Associated_set and c in Expressed_set],
        'non_assoc_silent': [coord_to_name[c] for c in coords_list if c not in Associated_set and c not in Expressed_set]
    }
    
    print(f"\n--- Feature Distribution for {base_name} ---")
    for cat, names in cat_map.items():
        print(f"  > {cat:15}: {len(names)} features")
    
    long_dfs = []
    for cat_label, feature_names in cat_map.items():
        if feature_names:
            sub_df = res_df[feature_names]
            long = sub_df.reset_index().melt(id_vars='index')
            long.columns = ['sample', 'feature', 'value']
            long['view'] = f"{base_name}_{cat_label}"
            long['group'] = 'single_group'
            long_dfs.append(long)
            
    return long_dfs

# Generate all categorical views
all_long_data = []

# 1. From Expression
all_long_data.extend(split_and_format(df_expr_res, expr_coords, "Expr"))

# 2. From Methylation Contexts
for context in meth_files.keys():
    all_long_data.extend(split_and_format(meth_residuals_master[context], meth_coords_master[context], f"m{context}"))

df_mofa = pd.concat(all_long_data)

print(f"MOFA data prepared with {df_mofa['sample'].nunique()} genotypes and {df_mofa['view'].nunique()} categorical views.")
print("Generated views:", df_mofa['view'].unique())

### --- 4. RUN MOFA ---
ent = entry_point()
ent.set_data_df(df_mofa)
ent.set_data_options(use_float32=True, scale_views=True)
ent.set_model_options(factors=MOFA_FACTORS, spikeslab_weights=True, ard_weights=True) #
ent.set_train_options(convergence_mode="slow", seed=1, verbose=False)

ent.build()
ent.run()
ent.save(f"{path_hdf5}{hdf5_file}")

### --- 5. PLOTS & STATS ---
m = mfx.mofa_model(f"{path_hdf5}{hdf5_file}")
# 1. Get ONLY factors that have variance
weights_all = m.get_weights(df=True)
active_factors = weights_all.columns[weights_all.var() > 1e-10].tolist()
final_weights = weights_all[active_factors]

#basics
print(f"Genotypes: {m.shape[0]}")
print(f"Views: {', '.join(m.views)}")
print(f"Factors trained: {m.nfactors}")
print(f"Active Factors with variance: {len(active_factors)}")

# A. R2 Statistics
# --- Updated Total Model Variance Calculation ---
r2_df = m.get_r2()
view_totals = r2_df.groupby('View')['R2'].sum() # 1. Total per View

# 2. Grand Total Calculation (Mean of all views)
# We ensure we are working with percentages (0-100)
view_totals_pct = view_totals.apply(lambda x: x if x > 1.0 else x * 100)
grand_total_variance = view_totals_pct.mean()

print(f"MEAN GRAND TOTAL VARIANCE EXPLAINED (ALL VIEWS): {grand_total_variance:.2f}%")
print(f"Number of views analyzed: {len(view_totals)}")
print("="*30)

# --- NEW: TOTAL VARIANCE PER FACTOR (Across all views) ---
print("\n" + "="*30 + "\nTOTAL VARIANCE EXPLAINED PER FACTOR\n" + "="*30)
# Summing R2 across views tells us the total 'units' of variance captured by the factor
# then we divide by the number of views to get the global average contribution
factor_totals = r2_df.groupby('Factor')['R2'].sum() / len(m.views)
factor_totals_pct = factor_totals.apply(lambda x: x if x > 1.0 else x * 100)

# Sort factors by their impact (highest variance first)
factor_totals_sorted = factor_totals_pct.sort_values(ascending=False)

for factor, total in factor_totals_sorted.items():
    if factor in active_factors:
        print(f" - {factor:10}: {total:.2f}% of total dataset variance")
print("="*30)

print("\n" + "="*30 + "\nTOTAL VARIANCE EXPLAINED PER VIEW\n" + "="*30)
view_totals = r2_df.groupby('View')['R2'].sum()
for view, total in view_totals.items():
    display_val = total if total > 1.0 else total * 100
    print(f" - {view}: {display_val:.2f}%")

# --- Variance Explained by Category ---
view_groups = {
    'mCG': [v for v in m.views if 'mCG' in v],
    'mCHG': [v for v in m.views if 'mCHG' in v],
    'mCHH': [v for v in m.views if 'mCHH' in v],
    'Expr': [v for v in m.views if 'Expr' in v]
}

print("MEAN TOTAL VARIANCE BY CONTEXT:")
for group_name, views in view_groups.items():
    if views:
        group_avg = view_totals[views].mean()
        # Adjust display if the raw R2 is a decimal
        display_avg = group_avg if group_avg > 1.0 else group_avg * 100
        print(f" - {group_name:5}: {display_avg:.2f}%")


# B. Pivot Table for stdout - Sorting by Methylation to see Factor 1 first
r2_pivot = r2_df.pivot(index='Factor', columns='View', values='R2')
print("\nR2 per Factor (Sorted by mCG_cis_only):")
# We sort by the second column (Methylation_Resid) to bring Factor 1 to the top
print(r2_pivot.sort_values(by='mCG_cis_only', ascending=False).head(len(active_factors)))
print("="*30 + "\n")

order_views_to_plot = [
    "mCG_cis_only", "mCG_cistrans", "mCG_trans_only", "mCG_non_assoc_expr", "mCG_non_assoc_silent", 
    "mCHG_cis_only", "mCHG_cistrans", "mCHG_trans_only", "mCHG_non_assoc_expr", "mCHG_non_assoc_silent", 
    "mCHH_cis_only", "mCHH_cistrans", "mCHH_trans_only", "mCHH_non_assoc_expr", "mCHH_non_assoc_silent", 
    "Expr_cis_only", "Expr_cistrans", "Expr_trans_only", "Expr_non_assoc_expr"
]

# C. PLOT VARIANCE EXPLAINED (R2) FROM SCRATCH
try:
    # 1. Get and process R2 data
    r2_data = m.get_r2()
    factor_sums = r2_data.groupby('Factor')['R2'].sum()
    active_factors_list = factor_sums[factor_sums > 1e-6].index.tolist()
    active_factors_list.sort(key=lambda x: int(x.replace('Factor', '')))

    # 2. Pivot and reindex
    r2_matrix = r2_data.pivot(index='Factor', columns='View', values='R2')
    r2_matrix = r2_matrix.reindex(index=active_factors_list, columns=order_views_to_plot)

    # 3. Increase figure size relative to the grid to make squares appear smaller
    # Wider figure + smaller 'shrink' value for colorbar = more refined squares
    fig, ax = plt.subplots(figsize=(22, 10))
    
    # 4. Create Annotation Mask
    # Only show numbers if R2 > 0.5% to keep the plot clean
    annot_labels = r2_matrix.applymap(lambda x: f'{x:.1f}' if x > 0.5 else "")

    # 5. Create Heatmap
    # Creates a palette from a light tint to your specific color
    custom_palette = sns.light_palette("#8A2BE2", as_cmap=True)
    sns.heatmap(
        r2_matrix, 
        annot=annot_labels,      # Use our custom formatted strings
        fmt="",                  # Required when passing a string matrix to annot
        annot_kws={"size": 9},   # Smaller font for the numbers inside squares
        cmap=custom_palette, #"Blues", 
        square=True,
        cbar_kws={
            'label': 'Variance Explained (%)', 
            'shrink': 0.4,       # Smaller colorbar makes the whole plot feel more refined
            'pad': 0.01
        },
        linewidths=2,            # Thicker lines effectively make the blue squares look smaller
        linecolor='white',
        ax=ax
    )

    # --- ADD VERTICAL SEPARATOR LINES HERE ---
    # 5 categories per context (mCG, mCHG, mCHH), the lines go after index 5, 10, and 15.
    for x_pos in [5, 10, 15]:
        ax.axvline(x=x_pos, color='black', lw=3, linestyle='-')

    # 6. ENHANCE FONTS AND PLACEMENT
    # X-axis (Views)
    ax.xaxis.tick_top() 
    ax.set_xticklabels(order_views_to_plot, rotation=90, fontsize=20, fontweight=700)
    ax.set_xlabel("") 

    # Y-axis (Factors)
    ax.set_yticklabels(active_factors_list, rotation=0, fontsize=20, fontweight=700)
    ax.set_ylabel("", fontsize=24, fontweight=700, labelpad=25)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.set_label('Variance Explained (%)', fontsize=18, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=14)

    # 7. Final Layout Adjustment
    # Use subplots_adjust to give more room to the top labels
    plt.subplots_adjust(top=0.8, bottom=0.1)

    # 8. Save
    plt.savefig(f"{path_hdf5}MOFA_AllContexts_R2_varianceexplained_RESIDUALS.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{path_hdf5}MOFA_AllContexts_R2_varianceexplained_RESIDUALS.svg", format="SVG", dpi=2000, bbox_inches='tight')
    print("Saved: MOFA_AllContexts_R2_varianceexplained.png with annotations and refined squares.")
    plt.close()

except Exception as e:
    print(f"Error plotting R2 from scratch: {e}")

# F. FACTOR DISTRIBUTION (Violin Plot) 
# y plots the "Centered Latent Factor Scores" 
try:
    factors_df = m.get_factors(df=True)
    # Use m.nfactors to ensure we plot all factors actually trained by the model, will help check if the last factor is truly "flat" (zero variance)
    df_plot = factors_df.iloc[:, :len(active_factors)].melt(var_name='Factor', value_name='Value')
    plt.figure(figsize=(12, 6))
    # inner='quartile' keeps the statistical summary inside the violin
    sns.violinplot(x='Factor', y='Value', data=df_plot, palette="light:steelblue", 
                   inner='quartile', cut=0)
    # Adding a stripplot on top helps see individual accession points
    sns.stripplot(x='Factor', y='Value', data=df_plot, size=1.5, color='black', alpha=0.2, jitter=True)
    plt.ylabel("Latent Factor Score (Sample Loadings)", fontsize=18, fontweight=700)
    plt.title(f"") # Distribution of {len(active_factors)} Latent Factors (All Accessions)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{path_hdf5}MOFA_AllContexts_Factor_Distributions_Violin_RESIDUALS.png", dpi=300)
    plt.savefig(f"{path_hdf5}MOFA_AllContexts_Factor_Distributions_Violin_RESIDUALS.svg", format="SVG", dpi=2000, bbox_inches='tight')
    plt.close()
    print(f"Saved: MOFA_AllContexts_Factor_Distributions_Violin.png")
except Exception as e:
    print(f"Error plotting distributions: {e}")

### --- G. MULTI-VIEW SCATTER LOOP (FACTORS 1-10) ---
try:
    # Target factors 1 through 10
    target_factors = [f"Factor{i}" for i in range(1, 11)] 

    for target_factor in target_factors:
        # 1. Get Factor Scores
        f_scores_raw = m.get_factors(factors=target_factor, df=True)
        f_scores_df = f_scores_raw.reset_index()
        f_scores_df.columns = ['sample_id', 'factor_score']
        
        # 2. Setup Grid
        n_views = len(order_views_to_plot)
        n_cols = 5
        n_rows = math.ceil(n_views / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5 * n_rows))
        axes_flat = axes.flatten()
        
        # 3. Iterate through Views
        for i, view in enumerate(order_views_to_plot):
            ax = axes_flat[i]
            
            # Get weights to find the top driver
            w = m.get_weights(views=view, df=True)
            top_feature = w.abs().sort_values(target_factor, ascending=False).index[0]
            
            # Get data for that feature
            view_data_raw = m.get_data(views=view, features=top_feature, df=True)
            view_data = view_data_raw.reset_index()
            
            # Identify sample ID and value columns
            potential_id_cols = [c for c in view_data.columns if c.lower() in ['sample', 'id', 'index']]
            id_col = potential_id_cols[0] if potential_id_cols else view_data.columns[0]
            val_col = top_feature 
            
            # Merge factors and data
            merged = view_data.merge(f_scores_df, left_on=id_col, right_on='sample_id')
            
            if not merged.empty:
                # Plot with 99% Confidence Interval
                sns.regplot(
                    x='factor_score', 
                    y=val_col, 
                    data=merged, 
                    ax=ax,
                    ci=99, 
                    scatter_kws={'alpha':0.3, 's':15, 'edgecolor':'none'}, 
                    line_kws={'color':'red', 'lw':2}
                )
                
                ax.set_title(f"{view}\n{top_feature}", fontsize=12, fontweight='bold')
                ax.set_xlabel(f"Score ({target_factor})", fontsize=10)
                ax.set_ylabel("Value", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No Overlapping Data", ha='center', fontsize=10)
                ax.set_title(f"{view}\n(Empty)", fontsize=12)

        # 4. Remove unused axes from the grid
        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.suptitle(f"Top Drivers per View: {target_factor} (99% CI)", fontsize=24, y=1.02, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{path_hdf5}MOFA_Interpreting_{target_factor}_RESIDUALS.png"
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved 99% CL interval: {save_path}")
except Exception as e:
    print(f"Error in Multi-View Factor Loop: {e}")

# H. WEIGHT COHERENCE HEATMAP
try:
    # 1. Compute correlation
    corr_matrix = final_weights.corr()
    # 2. Plotting
    plt.figure(figsize=(12, 10))
    # We assign the heatmap to 'ax' to access the colorbar
    ax = sns.heatmap(
        corr_matrix, 
        cmap="RdBu_r", 
        center=0, 
        annot=False, 
        square=True,
        cbar_kws={'shrink': 0.8} # Adjust colorbar size to match the square plot
    )
    # 3. ADD COLORBAR LABEL
    cbar = ax.collections[0].colorbar
    cbar.set_label('Pearson Correlation Coefficient (r)', fontsize=14, fontweight=700, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    # 4. ENHANCE TITLES AND AXES
    plt.title(f"", fontsize=16, fontweight=700, pad=20) #Coherence of {len(active_factors)} Active Factors
    plt.xticks(fontname='arial', fontsize=18, fontweight=700)
    plt.yticks(fontname='arial', fontsize=18, fontweight=700)
    plt.tight_layout()
    plt.savefig(f"{path_hdf5}MOFA_AllContexts_Weight_Coherence_Heatmap_RESIDUALS.png", dpi=300)
    plt.savefig(f"{path_hdf5}MOFA_AllContexts_Weight_Coherence_Heatmap_RESIDUALS.svg", format="SVG", dpi=2000, bbox_inches='tight')
    plt.close()
    print(f"Saved: Coherence Heatmap with label and {len(active_factors)} factors.")
except Exception as e:
    print(f"Error plotting weight heatmap: {e}")

### --- I. EXTRACT TOP FEATURES FOR ALL ACTIVE FACTORS ---
print(f"\n--- Extracting Top 5 UNIQUE Features and their Top-Weight Views ---")

top_summary_list = []

# Pre-calculate top features per view to make the "cross-check" faster
view_top_hits = {}
for view in m.views:
    # Get all weights for this view
    vw = m.get_weights(views=view, df=True)
    view_top_hits[view] = {}
    for factor in active_factors:
        # Identify features in the top 5 for THIS specific view and factor
        top_5_in_view = vw[factor].abs().sort_values(ascending=False).head(5).index.tolist()
        view_top_hits[view][factor] = set(top_5_in_view)

for factor in active_factors:
    all_weights_for_factor = []
    
    # 1. Collect weights from EVERY view
    for view in m.views:
        w = m.get_weights(views=view, factors=factor, df=True)
        for feature, weight in w[factor].items():
            all_weights_for_factor.append({
                'Feature': feature,
                'View': view,
                'Weight': weight,
                'Abs_Weight': abs(weight)
            })
    
    # 2. Sort by Absolute Weight (Global ranking for this factor)
    factor_df = pd.DataFrame(all_weights_for_factor).sort_values(by='Abs_Weight', ascending=False)
    
    # 3. Identify the Top 5 GLOBAL unique features
    seen_features = set()
    unique_top_5_features = []
    
    for _, row in factor_df.iterrows():
        if len(unique_top_5_features) >= 5:
            break
        if row['Feature'] not in seen_features:
            unique_top_5_features.append(row['Feature'])
            seen_features.add(row['Feature'])
    
    # 4. For these 5 features, find ALL views where they were in the Top 5
    print(f"\nTop 5 Unique Features for {factor} (with all active views):")
    for feat in unique_top_5_features:
        # Find which views have this feature in their top 5
        contributing_views = []
        for view in m.views:
            if feat in view_top_hits[view][factor]:
                # Get the weight for this specific view-feature-factor combo
                w_val = factor_df[(factor_df['View'] == view) & (factor_df['Feature'] == feat)]['Weight'].values[0]
                contributing_views.append(f"{view}[{w_val:.3f}]")
        
        views_str = " | ".join(contributing_views)
        print(f"  > {feat:25} | Top in Views: {views_str}")
        
        # Store for CSV (storing the views_str for the summary)
        top_summary_list.append({
            'Factor': factor,
            'Feature': feat,
            'Contributing_Views': views_str
        })

# --- J. EXPORT RECOGNIZED FEATURES TO CSV ---
try:
    final_df = pd.DataFrame(top_summary_list)
    
    # Save Detailed CSV
    detailed_path = f"{path_hdf5}MOFA_Top_Features_Detailed_RESIDUALS.csv"
    final_df.to_csv(detailed_path, index=False)
    
    # Save Summary CSV (Condensed format)
    summary_data = []
    for factor in active_factors:
        sub = final_df[final_df['Factor'] == factor]
        if not sub.empty:
            # Format: "TE1(Views...), TE2(Views...)"
            feature_list = " // ".join([f"{row['Feature']}({row['Contributing_Views']})" for _, row in sub.iterrows()])
            summary_data.append({'Factor': factor, 'Top_Features_Summary': feature_list})
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{path_hdf5}MOFA_Top_Features_Summary_RESIDUALS.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nFiles Saved Successfully:\n 1. {detailed_path}\n 2. {summary_path}")
except Exception as e:
    print(f"Could not save summary CSVs: {e}")

# J. EXPORT LATENT FACTORS
factors_df = m.get_factors(df=True)
factors_df.to_csv(f"{path_hdf5}Arabidopsis_MOFA_Factors_RESIDUALS.csv") # save to CSV

m.close()