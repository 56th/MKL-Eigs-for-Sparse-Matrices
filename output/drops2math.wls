#!/usr/bin/env wolframscript
(* ::Package:: *)

If[Length@$ScriptCommandLine < 4, Print["Error: please specify surface, FE, and meshlevel"]; Quit[]];
surface = $ScriptCommandLine[[2]];
FE = $ScriptCommandLine[[3]];
meshLevel = ToExpression@$ScriptCommandLine[[4]];


h = {0.833332,0.416666,0.208333,0.104167,0.052083,0.026042,0.013021};
hStr = ToString/@h;
Print["h = ", hStr[[meshLevel]]];
inDir = surface<>"/"<>FE<>"/blocks/h="<>hStr[[meshLevel]]<>"_";
inDirC = surface<>"/P1P1/blocks/h="<>hStr[[meshLevel]]<>"_";
oDir = surface<>"/"<>FE<>"/h="<>hStr[[meshLevel]]<>"_";
Print["input:  ", inDir, "*"];
Print["output: ", oDir, "*"];
Print[""];


A = Import[inDir<>"A.mtx"];
B = Import[inDir<>"B.mtx"];
M0 = Import[inDir<>"M.mtx"];
nA = Length@A;
nS = Length@M0;
Cfull = Import[inDirC<>"C_full.mtx"];
Cn = Import[inDirC<>"C_n.mtx"];
Print["size of A      = ", Length@A]
Print["size of M      = ", Length@M0]
Print["size of C_n    = ", Length@Cn]
Print["size of C_full = ", Length@Cfull]


eps = 10.^-6;
ABLOCK0 = ArrayFlatten[{{A, Transpose@B}, {B, 0.}}];
MBLOCK0 = ArrayFlatten[{{SparseArray[{{i_,i_} -> eps}, {nA, nA}](*eps A*), 0.}, {0., M0}}];
ABLOCKn = ArrayFlatten[{{A, Transpose@B}, {B, -Cn}}];
MBLOCKn = ArrayFlatten[{{SparseArray[{{i_,i_} -> eps}, {nA, nA}](*eps A*), 0.}, {0., M0 + Cn}}];
ABLOCKfull = ArrayFlatten[{{A, Transpose@B}, {B, -Cfull}}];
MBLOCKfull = ArrayFlatten[{{SparseArray[{{i_,i_} -> eps}, {nA, nA}](*eps A*), 0.}, {0., M0 + Cfull}}];
Print["max memory used:  ", ToString@TeXForm@ScientificForm[MaxMemoryUsed[] * 10.^-6, 3], " MB"];
Print["memory available: ", ToString@TeXForm@ScientificForm[MemoryAvailable[] * 10.^-6, 3], " MB"];
Print[""];


format = "mtx";
Print@Export[oDir<>"size.txt", <|"nA" -> nA, "nS" -> nS|>];
Print[Export[oDir<>"A_block_0."<>format, ABLOCK0] <> " " <> Export[oDir<>"eps="<>ToString@DecimalForm@eps<>"_M_block_0."<>format, MBLOCK0]];
Print[Export[oDir<>"A_block_n."<>format, ABLOCKn] <> " " <> Export[oDir<>"eps="<>ToString@DecimalForm@eps<>"_M_block_n."<>format, MBLOCKn]];
Print[Export[oDir<>"A_block_full."<>format, ABLOCKfull] <> " " <> Export[oDir<>"eps="<>ToString@DecimalForm@eps<>"_M_block_full."<>format, MBLOCKfull]];
