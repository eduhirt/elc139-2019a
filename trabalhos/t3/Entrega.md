# Scheduling com OpenMP

Nome: Eduardo Rafael Hirt


Implementaço: [OpenMPDemoABC](OpenMPDemoABC.cpp)

###Resultados:


**Dynamic com Chunk:**<br/>
ADACACCDABDACBDBDCBB<br/>
A=5 B=5 C=5 D=5 <br/>
Num total:20 <br/>
----------------------<br/>
**Dynamic sem Chunk:**<br/>
BCCBADCDACBCADADACDB<br/>
A=5 B=4 C=6 D=5 <br/>
Num total:20<br/>
----------------------<br/>
**Static com Chunk:**<br/>
BBBDABCBDADDDCACCCAA<br/>
A=5 B=5 C=5 D=5 <br/>
Num total:20<br/>
----------------------<br/>
**Static sem Chunk:**<br/>
DADDDCBABBBACABDCACC<br/>
A=5 B=5 C=5 D=5 <br/>
Num total:20<br/>
----------------------<br/>
**Guided com chunk:**<br/>
DAAAADCADBDDBCBCBCBC<br/>
A=5 B=5 C=5 D=5 <br/>
Num total:20<br/>
----------------------<br/>
**Guided sem chunk:**<br/>
BCCADCBCCADBCDACBDBA<br/>
A=4 B=5 C=7 D=4 <br/>
Num total:20<br/>
----------------------<br/>
**RunTime:**<br/>
DBCDBCDBADDACBACABAC<br/>
A=5 B=5 C=5 D=5 <br/>
Num total:20<br/>
----------------------<br/>
**Auto:**<br/>
DDBCDABAAAADCBDCBBCC<br/>
A=5 B=5 C=5 D=5 <br/>
Num total:20<br/>
----------------------<br/>
**Com Erro:**<br/>
AADCBADCBADCBADCB---<br/>
A=5 B=4 C=4 <br/>




