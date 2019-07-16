#!/usr/bin/env wolframscript
(* ::Package:: *)

If[Length@$ScriptCommandLine < 7, Print["Error: please specify surface, formulation, meshlevel, shift, [all|0|n|full], [mat|mtx]"]; Quit[]];
surface = $ScriptCommandLine[[2]];
form = $ScriptCommandLine[[3]];
meshLevel = ToExpression@$ScriptCommandLine[[4]];
shift = $ScriptCommandLine[[5]];
If[$ScriptCommandLine[[6]] != "all", stab = {$ScriptCommandLine[[6]]}, stab = {"0", "n", "full"}];
fmt = $ScriptCommandLine[[7]];


h = {0.833332,0.416666,0.208333,0.104167,0.052083,0.026042,0.013021};
hStr=If[StringLength@#<8,#<>StringJoin@Table["0",{8-StringLength@#}],#]&/@ToString/@h;
Print["h = ", hStr[[meshLevel]]];
inDir = "../../drops/output/NavierStokes_"<>surface<>"/test85_h="<>hStr[[meshLevel]]<>"/shift="<>shift<>"_form="<>form<>"_";
Print["input:  ", inDir, "*"];
Print[""];


Get["https://raw.githubusercontent.com/dih5/TgBot/master/TgBot/TgBot.m"];
Needs["TgBot`"]
tgChatID = Import["tg.txt", "List"][[1]];
tgToken = Import["tg.txt", "List"][[2]];
BotAPICall["getUpdates",{},{"Token"->tgToken}];
tg[text_, time_]:=BotAPICall["sendMessage", {"chat_id"->tgChatID, "text" -> 
	"eig.wls: " <> text <> " ("<> ToString@TeXForm@ScientificForm[time / 60., 3] <>" min, " <> ToString@TeXForm@ScientificForm[MaxMemoryUsed[] * 10.^-6, 3] <> " Mb)"},
{"Token"->tgToken}];


import[path_] := If[fmt == "mat", N@Import[path <> ".mat", "LabeledData"][[1, 2]], N@Import[path <> ".mtx", "MTX"]]


eps = 10.^-6;
time = First@AbsoluteTiming[
	A = import[inDir<>"A"];
	B = import[inDir<>"B"];
	M0 = import[inDir<>"M"];
	nA = Length@A;
	nS = Length@M0;
	Cfull = import[inDir<>"C_full"];
	Cn = import[inDir<>"C_n"];
	Print["size of A      = ", Length@A];
	Print["size of M      = ", Length@M0];
	Print["size of C_n    = ", Length@Cn];
	Print["size of C_full = ", Length@Cfull];
	Print["types: ", Head/@{A, B, M0, Cfull, Cn}];
	Print["max memory used:  ", ToString@TeXForm@ScientificForm[MaxMemoryUsed[] * 10.^-6, 3], " MB"];
	Print["memory available: ", ToString@TeXForm@ScientificForm[MemoryAvailable[] * 10.^-6, 3], " MB"];
	Print[""];
];
tg["import matrices (" <> inDir <> "*), n_A = " <> ToString@nA <> ", n_S = " <> ToString@nS, time];


findMinArn[A_, M_] := Module[{l, x},
	{l, x} = First@Transpose@Quiet@Eigensystem[{A, M}, -2, Method->{"Arnoldi"}];
	{-l, Norm[A.x - l(M.x)]}
]
findMaxArn[A_, M_] := Module[{l, x},
	{l, x} = First@Transpose@Quiet@Eigensystem[{A, M}, 1, Method->{"Arnoldi", "Shift"-> -5}];
	{-l, Norm[A.x - l(M.x)]}
]


Do[
	If[s == "0",
		a = ArrayFlatten[{{A, Transpose@B}, {B, 0.}}];
		b = ArrayFlatten[{{SparseArray[{{i_,i_} -> eps}, {nA, nA}], 0.}, {0., M0}}];
	];
	If[s == "n",
		a = ArrayFlatten[{{A, Transpose@B}, {B, -Cn}}];
		b = ArrayFlatten[{{SparseArray[{{i_,i_} -> eps}, {nA, nA}], 0.}, {0., M0 + Cn}}];
		Clear[Cn];
	];
	If[s == "full",
		a = ArrayFlatten[{{A, Transpose@B}, {B, -Cfull}}];
		b = ArrayFlatten[{{SparseArray[{{i_,i_} -> eps}, {nA, nA}], 0.}, {0., M0 + Cfull}}];
		Clear[Cfull];
	];
	Print["types of block matrices: ", Head/@{a, b}];
	time = First@AbsoluteTiming[
		{lMin, lMinRes} = findMinArn[a, b]; 
	];
	tg[stab <> "-stab:\nlambda_min = " <> ToString@TeXForm@ScientificForm[lMin, 3] <> ", residual = " <> ToString@TeXForm@ScientificForm[lMinRes, 3], time];
	time = First@AbsoluteTiming[
		{lMax, lMaxRes} = findMaxArn[a, b]; 
	];
	tg[s <> "-stab:\nlambda_max = " <> ToString@TeXForm@ScientificForm[lMax, 3] <> ", residual = " <> ToString@TeXForm@ScientificForm[lMaxRes, 3], time];
	Print[Export[inDir <> "stab=" <> s <> "_lambda.json", <| "n_A" -> nA, "n_S" -> nS, "lambdas" -> {lMin, lMax}, "residuals" -> {lMinRes, lMaxRes} |>]],
{s, stab}]
