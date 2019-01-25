#!/usr/bin/env wolframscript
(* ::Package:: *)

(* https://reference.wolfram.com/language/workflow/CreateWolframLanguageScripts.html *)
(* https://mathematica.stackexchange.com/a/175546/49280 *)
(* also may run w/: #!/usr/local/bin/wolfram -script *)
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
	"eigs.wls: " <> text <> " ("<> ToString@CForm[time / 60.] <>" min, " <> ToString@CForm[MemoryInUse[] * 10.^-6] <> " Mb)"},
{"Token"->tgToken}];


time = First@Timing[
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


time = First@Timing[
	{lMin, lMinRes} = findMinArn[A, M]; 
];
tg["lambda_min = " <> ToString@CForm@lMin <> ", residual = " <> ToString@CForm@lMinRes, time];


time = First@Timing[
	{lMax, lMaxRes} = findMaxArn[A, M]; 
];
tg["lambda_max = " <> ToString@CForm@lMax <> ", residual = " <> ToString@CForm@lMaxRes, time];


Export[oName, <| "lambdas" -> {lMin, lMax}, "residuals" -> {lMinRes, lMaxRes} |>];
