#!/usr/bin/env wolframscript
(* ::Package:: *)

If[Length@$ScriptCommandLine < 4, Print["Error: please specify surface, FE, and meshlevel"]; Quit[]];
surface=$ScriptCommandLine[[2]];
FE=$ScriptCommandLine[[3]];
meshLevel=ToExpression@$ScriptCommandLine[[4]];


h={0.833332,0.416666,0.208333,0.104167,0.052083,0.026042,0.013021};
hStr=ToString/@h;
Print["h = ", hStr[[meshLevel]]];
inDir = surface<>"/"<>FE<>"/blocks/h="<>hStr[[meshLevel]]<>"_";
inDirC = surface<>"/P1P1/blocks/h="<>hStr[[meshLevel]]<>"_";
oDir = surface<>"/"<>FE<>"/h="<>hStr[[meshLevel]]<>"_";


A=Import[inDir<>"A.mtx"];
B=Import[inDir<>"B.mtx"];
Subscript[M, 0]=Import[inDir<>"M.mtx"];
Subscript[n, A]=Length@A;
Subscript[n, S]=Length@Subscript[M, 0];
Subscript[C, full]=Import[inDirC<>"C_full.mtx"];
Subscript[C, n]=Import[inDirC<>"C_n.mtx"];
Print["size of A =      ", Length@A]
Print["size of M =      ", Length@Subscript[M, 0]]
Print["size of C_n    = ", Length@Subscript[C, n]]
Print["size of C_full = ", Length@Subscript[C, full]]


\[Epsilon] =10.^-6;
Subscript[\[ScriptCapitalA], 0]=ArrayFlatten[{{A,Transpose@B},{B,0.}}];
Subscript[\[ScriptCapitalM], 0]=ArrayFlatten[{{ \[Epsilon] A(*SparseArray[{{i_,i_}->\[Epsilon]},{Subscript[n, A],Subscript[n, A]}]*), 0.},{0.,Subscript[M, 0]}}];
Subscript[\[ScriptCapitalA], full]=ArrayFlatten[{{A,Transpose@B},{B,-Subscript[C, full]}}];
Subscript[\[ScriptCapitalM], full]=ArrayFlatten[{{ \[Epsilon] A(*SparseArray[{{i_,i_}->\[Epsilon]},{Subscript[n, A],Subscript[n, A]}]*),0.},{0.,Subscript[M, 0]+Subscript[C, full]}}];
Subscript[\[ScriptCapitalA], n]=ArrayFlatten[{{A,Transpose@B},{B,-Subscript[C, n]}}];
Subscript[\[ScriptCapitalM], n]=ArrayFlatten[{{ \[Epsilon] A(*SparseArray[{{i_,i_}->\[Epsilon]},{Subscript[n, A],Subscript[n, A]}]*),0.},{0.,Subscript[M, 0]+Subscript[C, n]}}];


format="mtx";
Print@Export[oDir<>"size.txt", <|"nA"->Subscript[n, A],"nS"->Subscript[n, S]|>]
Print[Export[oDir<>"A_block_0."<>format,Subscript[\[ScriptCapitalA], 0]]<>" "<>Export[oDir<>"eps="<>ToString@DecimalForm@\[Epsilon]<>"_M_block_0."<>format,Subscript[\[ScriptCapitalM], 0]]]
Print[Export[oDir<>"A_block_full."<>format,Subscript[\[ScriptCapitalA], full]]<>" "<>Export[oDir<>"eps="<>ToString@DecimalForm@\[Epsilon]<>"_M_block_full."<>format,Subscript[\[ScriptCapitalM], full]]]
Print[Export[oDir<>"A_block_n."<>format,Subscript[\[ScriptCapitalA], n]]<>" "<>Export[oDir<>"eps="<>ToString@DecimalForm@\[Epsilon]<>"_M_block_n."<>format,Subscript[\[ScriptCapitalM], n]]]
