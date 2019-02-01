#!/usr/bin/env wolframscript
(* ::Package:: *)

(* https://reference.wolfram.com/language/workflow/CreateWolframLanguageScripts.html *)
(* https://mathematica.stackexchange.com/a/175546/49280 *)
(* also may run w/: #!/usr/local/bin/wolfram -script *)
If[Length@$ScriptCommandLine < 3, Print["Error: please specify 2 matrices"]; Quit[]];
{aName, mName} = $ScriptCommandLine[[{2, 3}]];
oName = aName <> ".lambda.txt";
Print["input:  ", aName, "\n        ", mName];
Print["output: ", oName];


(* SetDirectory@NotebookDirectory[]
h = ToString@First@Import["h.txt", "List"];
aName = "P1P1/h="<>h<>"_A_block_0.mtx";
mName = "P1P1/h="<>h<>"_eps=0.0001_M_block_0.mtx"; *)


Get["https://raw.githubusercontent.com/dih5/TgBot/master/TgBot/TgBot.m"];
Needs["TgBot`"]
tgChatID = Import["tg.txt", "List"][[1]];
tgToken = Import["tg.txt", "List"][[2]];
BotAPICall["getUpdates",{},{"Token"->tgToken}];
tg[text_, time_]:=BotAPICall["sendMessage", {"chat_id"->tgChatID, "text" -> 
	"eig.wls: " <> text <> " ("<> ToString@TeXForm@ScientificForm[time / 60., 3] <>" min, " <> ToString@TeXForm@ScientificForm[MaxMemoryUsed[] * 10.^-6, 3] <> " Mb)"},
{"Token"->tgToken}];


time = First@AbsoluteTiming[
	A = Import[aName];
	M = Import[mName]; 
	n = Length@A;
];
tg["import matrices (" <> aName <> "), n = " <> ToString@n, time];


findMinArn[A_, M_] := Module[{l, x},
	{l, x} = First@Transpose@Eigensystem[{A, M}, -2, Method->{"Arnoldi"}];
	{-l, Norm[A.x - l(M.x)]}
]
findMaxArn[A_, M_] := Module[{l, x},
	{l, x} = First@Transpose@Eigensystem[{A, M}, 1, Method->{"Arnoldi", "Shift"-> -5}];
	{-l, Norm[A.x - l(M.x)]}
]


time = First@AbsoluteTiming[
	{lMin, lMinRes} = findMinArn[A, M]; 
];
tg["lambda_min = " <> ToString@TeXForm@ScientificForm[lMin, 3] <> ", residual = " <> ToString@TeXForm@ScientificForm[lMinRes, 3], time];


time = First@AbsoluteTiming[
	{lMax, lMaxRes} = findMaxArn[A, M]; 
];
tg["lambda_max = " <> ToString@TeXForm@ScientificForm[lMax, 3] <> ", residual = " <> ToString@TeXForm@ScientificForm[lMaxRes, 3], time];


Export[oName, <| "n" -> n, "lambdas" -> {lMin, lMax}, "residuals" -> {lMinRes, lMaxRes} |>];
