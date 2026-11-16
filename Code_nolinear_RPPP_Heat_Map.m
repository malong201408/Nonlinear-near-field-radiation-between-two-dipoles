(* ::Package:: *)

(* ============================================================================ *)
(*  ver41 -- SPEED + RESTARTABILITY fix for the integNL12 heat map.             *)
(*           Physics, constants, tensors, grids: unchanged from ver40.          *)
(*                                                                              *)
(*  WHY ver40 NEVER FINISHED (job 34026604, killed at 48 h inside the NL map):  *)
(*   - For identical dipoles the net integrand  Q1->2 - Q2->1  is the          *)
(*     difference of two nearly equal ZERO-POINT pieces.  Over most of the     *)
(*     v-range they cancel to ~1e-13 of their size, so what is left there is   *)
(*     round-off noise.  PrecisionGoal 8 / AccuracyGoal 40 / MaxErrorIncreases *)
(*     500 made NIntegrate integrate that noise.  In a Python emulation of    *)
(*     NIntegrate's adaptive loop, ~1 grid point in 8 never converged (~32000  *)
(*     evaluations each, vs ~500 typical) and was then thrown away by          *)
(*     Quiet@Check anyway.                                                     *)
(*   - nearWidths = 200 with etaRel = 0.01 put 512 of 515 w-rows on the        *)
(*     expensive settings (the log itself printed "99.% pay full cost").        *)
(*   - Row R = EffR2 (r2 = 1) is exactly the static polarization catastrophe  *)
(*     det(I - T a T a) = 0 at w -> 0: the v-integrand grows like 1/v there,   *)
(*     so that row is divergent and can never converge.                        *)
(*   - alphaTens1/2 were memoized, but they are called at every quadrature    *)
(*     node (w1, w2 never repeat) -> unbounded cache growth in every kernel.   *)
(*   - No checkpoint: 48 h of finished points were lost when the job died.    *)
(*   - LaunchKernels[] only uses the node the master runs on: the 2nd node of *)
(*     the 2-node allocation was idle.                                         *)
(*                                                                              *)
(*  WHAT ver41 CHANGES                                                          *)
(*   [G1] Thermal-only NL integrand.  With coth = 1 + 2n, every triple          *)
(*        correlator splits as  ppp(T) = ppp(0) + dppp(T), where dppp is built  *)
(*        from Bose occupations only (TripleKaTh).  For identical dipoles the   *)
(*        zero-point part of Q1->2 - Q2->1 vanishes identically, so the net     *)
(*        integrand equals the dppp-only one with NO cancellation noise.        *)
(*        A built-in self-test checks this on your machine before anything is  *)
(*        computed; if it fails (e.g. non-identical dipoles) the original      *)
(*        integNL12 is used automatically.                                      *)
(*   [G2] v <-> 1-v symmetry (w1 <-> w2): integrate v in (0,1/2) and double.    *)
(*        Also self-tested.                                                     *)
(*   [G3] Breakpoints at the thermal scales m kT/(hbar w0 w) (where the net     *)
(*        integrand actually lives) plus the old resonance nodes; one option    *)
(*        set everywhere (PrecisionGoal 6).                                     *)
(*   [G4] Rows on the polarization catastrophe are detected and skipped.       *)
(*   [G5] alphaTens memo removed (geoData memo kept).                           *)
(*   [G6] Checkpoint: every finished R-row is written to figures_noF1/NL_rows.  *)
(*        A rerun resumes where it stopped.  Rows can be split over a SLURM     *)
(*        job array (NL_TASK_ID / NL_NTASKS); NL_MODE=assemble builds the CSVs. *)
(*   [G7] Finished stages are skipped: Q-vs-R and the integL12 map run only if  *)
(*        their CSVs are missing (see RUN CONTROL).  QTotalList reuses          *)
(*        QL12List + QNL12List instead of recomputing both.                     *)
(*   Points whose NIntegrate emitted a warning are still dropped from the maps  *)
(*   (as in ver40) but are now listed in integNL12_flagged.csv.                 *)
(* ============================================================================ *)

(* ============================================================================ *)
(*  ver40 -- ver39 with ONE change: the two tensor contractions are rewritten   *)
(*           in flattened (matrix-product) form.                                *)
(*                                                                              *)
(*  [V40] ContractTermLeft / ContractTermRight now use Flatten level specs      *)
(*        instead of Table[Sum[...]].  Same signatures, same return values,     *)
(*        same physics -- pure reindexing, verified to ~1e-14 against the       *)
(*        ver39 forms and against Q12 notes Eq. (51).                           *)
(*        Nothing else in this file is altered: every other definition,         *)
(*        tolerance, grid and control-panel value is byte-identical to ver39.   *)
(*                                                                              *)
(*        Run ver40_contraction_check.m once to confirm on your machine.        *)
(* ============================================================================ *)

(* ============================================================================ *)
(*  ver38 -- audit of integNL12 against Q12 notes (Eqs. 20-43) and SI-ver2_7    *)
(*                                                                              *)
(*  [F1] A11, A21, B11, B21 must use P21^{-1} = (I - T21 a1 T12 a2)^{-1}         *)
(*       (notes Eqs. 20, 22, 26, 28). The old code used P12^{-1} everywhere.     *)
(*       Identical for two identical, identically oriented dipoles only.        *)
(*       This also fixes integL12, which calls A11/A21.                          *)
(*  [F2] ContractTermLeft/Right conjugate their last argument themselves, but   *)
(*       were given A^dagger / B^dagger  -> double dagger. Now A and B are passed.*)
(*  [F3] each particle's fluctuations at its OWN temperature (p1 at T1, p2 at   *)
(*       T2), as in integL12. Old code used p2 at T1 and p1 at T2 in 4 terms.   *)
(*  [F4] triple correlator: symmetrized closed form from SI-ver2_7 Eq. (65).   *)
(*                                                                              *)
(*       Prefactor 1/(6 Pi^2)/(2 Pi) -> 1/(2 Pi).                               *)
(*  [F5] (not applied) w1 is kept in (0,w) as in the original code.            *)
(* ============================================================================ *)

(* ============================================================================ *)
(*  Code_nolinear_ver5_export_only.m   (revised from ..._ver32_..._rev_ver5)    *)
(*                                                                              *)
(*  RUN:   wolframscript -file Code_nolinear_ver5_export_only.m                 *)
(*    or:  Get["Code_nolinear_ver5_export_only.m"]  from a notebook / kernel    *)
(*                                                                              *)
(*  DATA-EXPORT-ONLY build:  ALL plotting/figure code removed.  Writes only CSV *)
(*  files into a ./figures subdirectory:                                        *)
(*    Q_vs_R_series.csv , Q_vs_R_axes.csv                                       *)
(*    integL12_XYZ.csv , integL12_matrix_signedlog.csv , integL12_matrix_raw.csv*)
(*    integNL12_XYZ.csv , integNL12_matrix_signedlog.csv                        *)
(*    integL12_colorbar.csv , integNL12_colorbar.csv                            *)
(*                                                                              *)
(*  SHARED COLORBAR:  integL12 and integNL12 use ONE common signed-log scale    *)
(*  set by cFloorGlobal / zmaxGlobal in the GLOBAL COLORBAR CONTROL PANEL near   *)
(*  the top of the file.  Their 'signedlog' columns are therefore directly      *)
(*  comparable; set Origin's color range to [-zmaxGlobal, +zmaxGlobal] for both.*)
(* ============================================================================ *)

(* ================================================================*)
(*Code_nonlinear_ver8.m*)
(*Nonlinear Casimir-Polder heat-transfer rate Q12=Q(T2)-Q(T1)*)
(**)(*Key fixes vs ver7:*)
(*The code is strictly following note_5_18*)
(*-Matrices order and off-diagonal terms is considered in AA and BB matrices*)
(*-Jacobian in the compactified integral corrected.*)(**)

(* ================================================================*)

ClearSystemCache[]
ClearAll["Global`*"];


(*----EXPORT SETUP:evaluate once----*)

(* --- robust base directory: works in notebook AND wolframscript --- *)
baseDir = Quiet@Check[
   If[$InputFileName =!= "", DirectoryName[$InputFileName], NotebookDirectory[]],
   Directory[]];
If[!StringQ[baseDir] || baseDir === "", baseDir = Directory[]];
outDir=FileNameJoin[{baseDir,"figures_noF1"}];
If[!DirectoryQ[outDir],CreateDirectory[outDir]];

(* NOTE: all figure/plot generation removed - this version ONLY exports CSV data. *)

Print["output directory: ",outDir];

(* ============================================================================ *)
(* #####                  [ver41]  RUN CONTROL                            ###### *)
(* ---------------------------------------------------------------------------- *)
(*  runQvsR / runL12Map :  Automatic = run only if its CSV is missing in outDir *)
(*                         (and only in NL_MODE=all).  True = always, False =   *)
(*                         never.  Set True if you change physics parameters.   *)
(*  Environment variables (set by the SLURM scripts; all optional):             *)
(*    NL_MODE    all (default) | compute | assemble                             *)
(*    NL_TASK_ID , NL_NTASKS   this task's index and the number of tasks that   *)
(*                             share the R-rows (round-robin)                   *)
(*    NL_KERNELS               number of subkernels (default: SLURM CPUs)       *)
(* ============================================================================ *)
runQvsR   = Automatic;
runL12Map = Automatic;

envOr[name_String, default_String] := With[{e = Environment[name]},
   If[StringQ[e] && StringLength[StringTrim[e]] > 0, StringTrim[e], default]];
envInt[names_List, default_Integer] := Module[{s, x},
   s = SelectFirst[Environment /@ names, StringQ[#] && StringMatchQ[StringTrim[#], DigitCharacter ..] &, None];
   x = If[s === None, default, ToExpression[StringTrim[s]]];
   If[IntegerQ[x] && x >= 0, x, default]];

nlMode   = ToLowerCase[envOr["NL_MODE", "all"]];
nlTaskID = envInt[{"NL_TASK_ID", "SLURM_ARRAY_TASK_ID"}, 0];
nlNTasks = Max[1, envInt[{"NL_NTASKS", "SLURM_ARRAY_TASK_COUNT"}, 1]];
nKernelsWanted = Max[1, envInt[{"NL_KERNELS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"}, $ProcessorCount]];
If[!MemberQ[{"all", "compute", "assemble"}, nlMode],
   Print["*** unknown NL_MODE = ", nlMode, " -> using all"]; nlMode = "all"];
If[nlTaskID >= nlNTasks,
   Print["*** NL_TASK_ID = ", nlTaskID, " >= NL_NTASKS = ", nlNTasks, " -> using ", Mod[nlTaskID, nlNTasks]];
   nlTaskID = Mod[nlTaskID, nlNTasks]];

doQvsR   = Switch[runQvsR, True, True, False, False, _,
   nlMode === "all" && !FileExistsQ[FileNameJoin[{outDir, "Q_vs_R_series.csv"}]]];
doL12Map = Switch[runL12Map, True, True, False, False, _,
   nlMode === "all" && !FileExistsQ[FileNameJoin[{outDir, "integL12_XYZ.csv"}]]];

ensureKernels[] := If[Length[Kernels[]] == 0,
   LaunchKernels[nKernelsWanted];
   Print[DateString[], "  launched ", Length[Kernels[]], " subkernels (requested ", nKernelsWanted, ")"]];

Print[DateString[], "  [ver41] NL_MODE = ", nlMode, "   task ", nlTaskID, " of ", nlNTasks,
      "   kernels wanted = ", nKernelsWanted];
Print["  stage Q-vs-R      : ", If[doQvsR, "RUN", "skip"]];
Print["  stage integL12 map: ", If[doL12Map, "RUN", "skip"]];
Print["  stage integNL12   : ", Switch[nlMode, "all", "compute + assemble", "compute", "compute only", "assemble", "assemble only"]];


(* ============================================================================ *)
(* #####            GLOBAL COLORBAR CONTROL PANEL  (adjust here)         ###### *)
(* ---------------------------------------------------------------------------- *)
(*  These two numbers set ONE common signed-log color scale shared by BOTH      *)
(*  the integL12 and integNL12 heat maps, so their exported 'signedlog'         *)
(*  columns are directly comparable and use an IDENTICAL colorbar in Origin.    *)
(*                                                                              *)
(*    signedlog(v) = sign(v) * log10( 1 + |v| / cFloorGlobal )                  *)
(*    colorbar range to set in Origin  ->  [ -zmaxGlobal , +zmaxGlobal ]        *)
(*                                                                              *)
(*  cFloorGlobal : absolute floor (NOT tied to each map's own vAbsMax).         *)
(*  zmaxGlobal   : absolute color-axis limit (NOT tied to each map's own peak). *)
(*                 Must exceed the largest |signedlog| or the peak saturates.   *)
(*                 With cFloorGlobal = 1e-22 the largest across the maps is      *)
(*                 ~19.5, so 20 is a safe round value.                          *)
(* ============================================================================ *)
cFloorGlobal = 1.*^-22;   (* shared absolute signed-log floor  *)
zmaxGlobal   = 20;        (* shared absolute color-axis limit  *)

(* ================================================================*)
(*SECTION 1:Physical constants*)
(* ================================================================*)

hbar=6.582119569*^-16;(*eV s*)
kB=8.6173303*^-5;(*eV/K*)
c=3.0*^18;(*Angstrom/s*)
eps0=1/(4 Pi);(*CGS units*)
(**both eps0SI and ESU are needed to fix the dimension mismatch between code and the paper **)
eps0SI=0.005526349406;(*e/(V A)*)
ESU=2.99792458*10^-6;(*V/A*)
T1=50;(*K,cold body*)
T2=400;(*K,hot body*)

(* ================================================================*)
(*SECTION 2:Molecular parameters *)
(* ================================================================*)

(* dipole 1 and 2 can not be the same *)
(*alpha tensor*)
a1xx=53;(*Angstrom^3*)
a1yy=0;
a1xy=0;
a1yx=a1xy;(*axy=ayx* in the calcuation*)

a2xx=53;(*Angstrom^3*)
a2yy=0;
a2xy=0;
a2yx=a2xy;(*axy=ayx* in the calcuation*)

alphaTensor1={{a1xx,a1xy,0},{a1yx,a1yy,0},{0,0,0}};
alphaTensor2={{a2xx,a2xy,0},{a2yx,a2yy,0},{0,0,0}};

alphaTensor1//MatrixForm
alphaTensor2//MatrixForm

(*beta tensor*)

(*beta tensor construction and symmetry rules*)
(*Indices:1=x,2=y,3=z*)
beta=Array[b,{3,3,3}];

(*beta values MUST put into accending order i.e.[yxx]->[xxy]*)
betaValues1={
b[1,1,1]->-250,(*\[Beta]_xxx*)
b[1,1,2]->0,(*\[Beta]_yxx*)
b[2,2,2]->0,(*\[Beta]_yyy*)
b[1,2,2]->0 (*\[Beta]_xyy*)};

betaValues2={
b[1,1,1]->-250,(*\[Beta]_xxx*)
b[1,1,2]->0,(*\[Beta]_yxx*)
b[2,2,2]->0,(*\[Beta]_yyy*)
b[1,2,2]->0  (*\[Beta]_xyy*)};
(*betaValues2={
b[3,3,3]->-17.5,(*\[Beta]_xxx*)
b[2,2,3]->0.3(*\[Beta]_yxx*)
};
*)

betaTensor1=Array[
Function[{i,j,k},
b@@Sort[{i,j,k}]/.betaValues1/. b[__]->0],
{3,3,3}];

betaTensor2=Array[
Function[{i,j,k},
b@@Sort[{i,j,k}]/. betaValues2/. b[__]->0],
{3,3,3}];

betaTensor1
betaTensor2

(* ============================================================ *)
(*  ROTATION OF alpha (rank-2) AND beta (rank-3) TENSORS         *)
(*  Single angle theta, rotation in the x-y plane.              *)
(*  It overwrites alphaTensor1/2 and betaTensor1/2 with their   *)
(*  rotated versions, so EVERYTHING downstream automatically    *)
(*  uses the rotated tensors with no other change needed.       *)
(* ============================================================ *)

(* ------------------------------------------------------------ *)
(*  ROTATION ANGLE  (edit this)                                 *)
(*  theta is in radians. Rotation is about the z-axis, i.e.     *)
(*  within the x-y plane. theta = 0 leaves tensors unchanged.   *)
(* ------------------------------------------------------------ *)

theta = 0;   (* e.g. Pi/6, Pi/4, Pi/2, ... *)

(* ------------------------------------------------------------ *)
(*  3x3 rotation matrix about z (rotation in the x-y plane)     *)
(* ------------------------------------------------------------ *)

rotMat[th_] := {
  {Cos[th], -Sin[th], 0},
  {Sin[th],  Cos[th], 0},
  {0,        0,       1}
};

(* ------------------------------------------------------------ *)
(*  Tensor rotation helpers                                     *)
(*                                                              *)
(*  Rank-2 (alpha):  a'_ij  = R_ip R_jq a_pq                    *)
(*                   = R . a . Transpose[R]                     *)
(*                                                              *)
(*  Rank-3 (beta):   b'_ijk = R_ip R_jq R_kr b_pqr             *)
(*  Implemented with TensorContract / explicit Sum so it works  *)
(*  for the full 3x3x3 array.                                   *)
(* ------------------------------------------------------------ *)

rotateRank2[a_, th_] := Module[{R = rotMat[th]},
  R . a . Transpose[R]
];

rotateRank3[b_, th_] := Module[{R = rotMat[th]},
  Table[
    Sum[
      R[[i, p]] R[[j, q]] R[[k, r]] b[[p, q, r]],
      {p, 3}, {q, 3}, {r, 3}
    ],
    {i, 3}, {j, 3}, {k, 3}
  ]
];

(* ------------------------------------------------------------ *)
(*  Keep an UNROTATED copy, then overwrite with rotated tensors *)
(*  (Storing the originals lets you re-run this cell with a     *)
(*   different theta without re-evaluating the definitions.)    *)
(* ------------------------------------------------------------ *)

If[! ValueQ[alphaTensor1Orig],
  alphaTensor1Orig = alphaTensor1;
  alphaTensor2Orig = alphaTensor2;
  betaTensor1Orig  = betaTensor1;
  betaTensor2Orig  = betaTensor2;
];

alphaTensor1 = rotateRank2[alphaTensor1Orig, theta];
alphaTensor2 = rotateRank2[alphaTensor2Orig, theta];
betaTensor1  = rotateRank3[betaTensor1Orig,  theta];
betaTensor2  = rotateRank3[betaTensor2Orig,  theta];

(* ------------------------------------------------------------ *)
(*  Quick sanity check (optional \[LongDash] comment out if not wanted)   *)
(* ------------------------------------------------------------ *)

Print["theta = ", theta, "  (", N[theta 180/Pi], " deg)"];
Print["Rotated alphaTensor1 = ", MatrixForm[Chop[alphaTensor1]//N]];
Print["Rotated alphaTensor2 = ", MatrixForm[Chop[alphaTensor2]//N]];
Print["Rotated betaTensor1  = ", betaTensor1//N];
Print["Rotated betaTensor2  = ", betaTensor2//N];
Print["betaTensor1 nonzero count: ",
      Count[Flatten[Chop[betaTensor1]], x_ /; x != 0]];
Print["betaTensor2 nonzero count: ",
      Count[Flatten[Chop[betaTensor2]], x_ /; x != 0]];

(* ============================================================ *)
(*  NOTE ON THE SYMMETRY ASSIGNMENT a2yx = a2xy                 *)
(*                                                              *)
(*  In the original notebook a1yx/a2yx are set equal to        *)
(*  a1xy/a2xy BEFORE alphaTensor is built. After rotation the   *)
(*  off-diagonal entries mix, so that equality no longer holds  *)
(*  in the rotated frame \[LongDash] which is physically correct: a       *)
(*  rotated tensor generally has a'_xy != a'_yx unless the      *)
(*  original was symmetric. If you specifically NEED the        *)
(*  rotated tensor to stay symmetric, the ORIGINAL must be      *)
(*  symmetric (a_xy = a_yx); rotation then preserves symmetry.  *)
(* ============================================================ *)

(*anisotropic parameter for dipole 1 and 2*)

(*xi1=1-a1yy/a1xx;
xi2=1-a2yy/a2xx;*)
xi1=0;
xi2=0;

(*characteristic frequencies \[Omega]0 for dipole 1 and 2*)
om001=2.9/hbar;(*rad/s*)
om002=2.9/hbar;(*rad/s*)
omR=om002/om001;
om01=1;(*rad/s*)
om02=omR;(*rad/s*)

(*Effective radius R and normalized parameter B for dipole 1 and 2*)
EffR1=(a1xx/(2*\[Pi]*eps0))^(1/3);(*Angstrom*)
EffB1=(4*\[Pi]*eps0SI)^(1/2)*(a1xx*(1-xi1))^(3/2)/(hbar*om001*om01)^(1/2);

EffR2=(a2xx/(2*\[Pi]*eps0))^(1/3);(*Angstrom*)
EffB2=(4*\[Pi]*eps0SI)^(1/2)*(a2xx*(1-xi2))^(3/2)/(hbar*om001*om02)^(1/2);

(*Normalized alpha and beta tensor for dipole 1 and 2*)

Atilde1=alphaTensor1/((2*\[Pi]*eps0)*(EffR1^3));
Atilde2=alphaTensor2/((2*\[Pi]*eps0)*(EffR2^3));

Btilde1=ESU*betaTensor1/EffB1;
Btilde2=ESU*betaTensor2/EffB2;

(*dimensionless reference wave-vector for dipole 1 and 2*)

k01=om001*om01*EffR1/c;
k02=om001*om02*EffR2/c;

(* ================================================================*)
(*SECTION 3:Scalar Lorentzians*)(**)(*All tensor quantities factor as (scalar) x (fixed tensor).*)(*lor[w] is the only frequency-dependent piece.*)(* ================================================================*)
eta1=0.015*om001;
eta2=0.015*om002;

lor1[w_]:=1/(1-w^2-I*eta1*(w/om001*om01));
lor2[w_]:=1/(1-w^2-I*eta2*(w/om001*om02));

(*Full polarisability tensor*)
alphaTens1[w_]:=Atilde1*lor1[w];
alphaTens2[w_]:=Atilde2*lor2[w];

(*Hyperpolarisability scalars *)
s1B12[w_,w1_]:=Btilde1*lor1[w]*lor1[w1]*lor1[w-w1]; (*w1,w2*)
s1B2N0[w_,w1_]:=Btilde1*lor1[-w1]*lor1[w-w1]*lor1[-w];(*w2,-w*)
s1B1N0[w_,w1_]:=Btilde1*lor1[-(w-w1)]*lor1[w1]*lor1[-w];(*w1,-w*)

s2B12[w_,w1_]:=Btilde2*lor2[w]*lor2[w1]*lor2[w-w1];
s2B2N0[w_,w1_]:=Btilde2*lor2[-w1]*lor2[w-w1]*lor2[-w];
s2B1N0[w_,w1_]:=Btilde2*lor2[-(w-w1)]*lor2[w1]*lor2[-w];

(* ================================================================*)
(*SECTION 4:Geometry (separation-dependent,frequency-free)*)
(* ================================================================*)

r1n={0,0,0};
r2n[r2_?NumericQ]:={r2,0,0};

(*Returns {Rn,RRmat}-- call once per r2,reuse for all frequencies*)
(*Note that r1 and r2 are  the normalized distance*)
(*G12=G21 so that they share the same geometry tensor*)
geoData[r2_?NumericQ]:=Module[{Rvec,Rn,Rhat},Rvec=r1n-r2n[r2];
Rn=Norm[Rvec];
Rhat=Rvec/Rn;
{Rn,Outer[Times,Rhat,Rhat]}];

(* ================================================================*)
(*SECTION 5:Free-space Interaction tensor T0(w) (normalized)*)
(* ================================================================*)

(*Note that Tn12 and Tn21 are different only on dimensionless wave-vector k01 and k02*)
(*8 instead of 4 to normalized with respect to 1/R^3 not 2/R^3*)
Tn12[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{kn1,knRn1,phase1,II},
kn1=k01*w;(*real*)
knRn1=kn1*Rn;(*real*)
phase1=Exp[I*knRn1];(*complex,magnitude 1*)
II=IdentityMatrix[3];
(phase1/(8*\[Pi]*eps0*Rn^3))*((knRn1^2+(I*knRn1-1))*II+(3*(1-I*knRn1)-knRn1^2)*RRmat)];

Tn21[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{kn2,knRn2,phase2,II},
kn2=k02*w;(*real*)
knRn2=kn2*Rn;(*real*)
phase2=Exp[I*knRn2];(*complex,magnitude 1*)
II=IdentityMatrix[3];
(phase2/(8*\[Pi]*eps0*Rn^3))*((knRn2^2+(I*knRn2-1))*II+(3*(1-I*knRn2)-knRn2^2)*RRmat)];

(* ================================================================*)
(*SECTION 6:P^{-1}(w) different by P12 and P21*)
(* ================================================================*)

(* P12 = I - T12 a2 T21 a1 ,  P21 = I - T21 a1 T12 a2   (notes, below Eq. 23) *)
Pinv[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{Tens12,Al2,Tens21,Al1},
Tens12=Tn12[w,Rn,RRmat];
Al2=alphaTens2[w];
Tens21=Tn21[w,Rn,RRmat];
Al1=alphaTens1[w];
Inverse[IdentityMatrix[3]-Tens12 . Al2 . Tens21 . Al1]];   (* = P12^{-1} *)

(* [F1 OFF] alias: P21^{-1} -> P12^{-1}, restoring the pre-F1 behavior.      *)
(* Valid only for identical, identically oriented dipoles.                   *)
Pinv21[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Pinv[w,Rn,RRmat];


(* ================================================================*)
(*SECTION 6:A11 A12 A21 A22*)
(* ================================================================*)
A11[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{Tens12,Al2,PinM,Tens21},
Tens12=Tn12[w,Rn,RRmat];
Al2=alphaTens2[w];
PinM=Pinv21[w,Rn,RRmat];   (* [F1] notes Eq. 20 *)
Tens21=Tn21[w,Rn,RRmat];
Tens12 . Al2 . PinM . Tens21];

A12[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{Tens12,PinM},
Tens12=Tn12[w,Rn,RRmat];
PinM=Pinv[w,Rn,RRmat];
PinM . Tens12];

A21[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{Tens21,PinM},
Tens21=Tn21[w,Rn,RRmat];
PinM=Pinv21[w,Rn,RRmat];   (* [F1] notes Eq. 22 *)
PinM . Tens21];

A22[w_?NumericQ,Rn_?NumericQ,RRmat_List]:=Module[{Tens21,Al1,PinM,Tens12},
Tens21=Tn21[w,Rn,RRmat];
Al1=alphaTens1[w];
PinM=Pinv[w,Rn,RRmat];
Tens12=Tn12[w,Rn,RRmat];
Tens21 . Al1 . PinM . Tens12];


(* ================================================================*)
(*SECTION 7:M111, M112, M221, M222*)
(* ================================================================*)

(*M tensors:beta tensor contracted with A11 A12 A21 A22 *)

M111[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,beta1, Atens11om1, Atens11om2},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
beta1=s1B12[w,w1];
Atens11om1=A11[w1,Rn,RRmat];
Atens11om2=A11[w2,Rn,RRmat];
TensorContract[TensorProduct[beta1,Atens11om1,Atens11om2],{{2,4},{3,6}}]];

M112[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,beta1, Atens12om1, Atens12om2},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
beta1=s1B12[w,w1];
Atens12om1=A12[w1,Rn,RRmat];
Atens12om2=A12[w2,Rn,RRmat];
TensorContract[TensorProduct[beta1,Atens12om1,Atens12om2],{{2,4},{3,6}}]];

M221[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,beta2, Atens21om1, Atens21om2},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
beta2=s2B12[w,w1];
Atens21om1=A21[w1,Rn,RRmat];
Atens21om2=A21[w2,Rn,RRmat];
TensorContract[TensorProduct[beta2,Atens21om1,Atens21om2],{{2,4},{3,6}}]];

M222[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,beta2, Atens22om1, Atens22om2},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
beta2=s2B12[w,w1];
Atens22om1=A22[w1,Rn,RRmat];
Atens22om2=A22[w2,Rn,RRmat];
TensorContract[TensorProduct[beta2,Atens22om1,Atens22om2],{{2,4},{3,6}}]];


(* ================================================================*)
(*SECTION 8: B tensors: B11, B12, B21, B22*)
(* ================================================================*)
B11[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,Tens12, Tens21, Al1,Al2,  PinvM,Mtens221, Mtens111},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
(*T matrix*)
Tens12=Tn12[w,Rn,RRmat];
Tens21=Tn21[w,Rn,RRmat];
(*alpha matrix*)
Al1=alphaTens1[w];
Al2=alphaTens2[w];
(*Pinv matrix*)
 PinvM=Pinv[w,Rn,RRmat];
(*M matrix*)
Mtens221=M221[r2,w,w1];
Mtens111=M111[r2,w,w1];
Tens12 . Al2 . Pinv21[w,Rn,RRmat] .   (* [F1] notes Eq. 26 *)
 (Tens21 . Al1 . Tens12 . Mtens221+Tens21 . Mtens111)+Tens12 . Mtens221];

B12[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,Tens12, Tens21, Al2,  PinvM,Mtens112, Mtens222},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
(*T matrix*)
Tens12=Tn12[w,Rn,RRmat];
Tens21=Tn21[w,Rn,RRmat];
(*alpha matrix*)
Al2=alphaTens2[w];
(*Pinv matrix*)
 PinvM=Pinv[w,Rn,RRmat];
(*M matrix*)
Mtens112=M112[r2,w,w1];
Mtens222=M222[r2,w,w1];
PinvM . (Tens12 . Al2 . Tens21 . Mtens112+Tens12 . Mtens222)];

B21[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,Tens12, Tens21, Al1,  PinvM,Mtens221, Mtens111},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
(*T matrix*)
Tens12=Tn12[w,Rn,RRmat];
Tens21=Tn21[w,Rn,RRmat];
(*alpha matrix*)
Al1=alphaTens1[w];
(*Pinv matrix*)
 PinvM=Pinv[w,Rn,RRmat];
(*M matrix*)
Mtens221=M221[r2,w,w1];
Mtens111=M111[r2,w,w1];
Pinv21[w,Rn,RRmat] . (   (* [F1] notes Eq. 28 *)
Tens21 . Al1 . Tens12 . Mtens221+Tens21 . Mtens111)];

B22[r2_?NumericQ,w_?NumericQ,w1_?NumericQ]:=Module[{w2,Rn,RRmat,Tens12, Tens21, Al1,Al2,  PinvM,Mtens112, Mtens222},
w2=w-w1;
{Rn,RRmat}=geoData[r2];
(*T matrix*)
Tens12=Tn12[w,Rn,RRmat];
Tens21=Tn21[w,Rn,RRmat];
(*alpha matrix*)
Al1=alphaTens1[w];
Al2=alphaTens2[w];
(*Pinv matrix*)
 PinvM=Pinv[w,Rn,RRmat];
(*M matrix*)
Mtens112=M112[r2,w,w1];
Mtens222=M222[r2,w,w1];
Tens21 . Al1 . PinvM . (Tens12 . Al2 . Tens21 . Mtens112+Tens12 . Mtens222)+Tens21 . Mtens112];

(* ================================================================*)
(*SECTION 9:Core integrand for linear heat transfer*)
(*Note that this is only for radiation from particle 1 to particle 2*)
(*The exchange power should be the substraction between integLinear1 and integLinear2*)
(* ================================================================*)
CothSafe[x_?NumericQ]:=If[Abs[x]<1*^-8,1/x,Coth[x]];
(*CothSafe[x_]:=If[Abs[x]<1*^-10,ComplexInfinity,Coth[x]];*)

integL12[r2_?NumericQ,w_?NumericQ,temp1_?NumericQ,temp2_?NumericQ]:=Module[{Rn,RRmat,Al1, Al2,  Atens11,Atens12,Atens21,Atens22, DagAtens11,DagAtens12,DagAtens21, DagAtens22, ImAl1,ImAl2, II,Ggeo11,Ggeo12, Ggeo21,Ggeo22},
{Rn,RRmat}=geoData[r2];
Al1=alphaTens1[w];
Al2=alphaTens2[w];

Atens11=A11[w,Rn,RRmat];
Atens12=A12[w,Rn,RRmat];
Atens21=A21[w,Rn,RRmat];
Atens22=A22[w,Rn,RRmat];

DagAtens11=Conjugate[Transpose[Atens11,{2,1}]];
DagAtens12=Conjugate[Transpose[Atens12,{2,1}]];
DagAtens21=Conjugate[Transpose[Atens21,{2,1}]];
DagAtens22=Conjugate[Transpose[Atens22,{2,1}]];

ImAl1=Im[Al1];
ImAl2=Im[Al2];
(*PP1=2*\[Pi]*ImAl1*Coth[(((hbar*om001*om01*w))/(2*(kB*temp1)))];*)(*linear fluctuation-dissipation theorem*)(*(((2)/(Exp[(((hbar*om01*w))/((kB*temp1)))]))+1)*)(*Coth[(((hbar*om01*w))/(2*(kB*temp1)))]*)
(*PP2=2*\[Pi]*ImAl2*Coth[(((hbar*om001*om02*w))/(2*(kB*temp2)))];*)(*linear fluctuation-dissipation theorem*)(*(((2)/(Exp[(((hbar*om02*w))/((kB*temp2)))]))+1)*)(*Coth[(((hbar*om02*w))/(2*(kB*temp2)))]*)
II=IdentityMatrix[3];

(*terms wrt T1 and T2 are regrouped and substracted in between QL1->2 and QL2->1*)
Ggeo11=om01^2*Tr[Im[(II+Al1 . Atens11) . ImAl1 . DagAtens11]];
Ggeo12=om01^2*Tr[Im[(Al1 . Atens12) . ImAl2 . DagAtens12]];
Ggeo21=om02^2*Tr[Im[(II+Al2 . Atens22) . ImAl2 . DagAtens22]];
Ggeo22=om02^2*Tr[Im[(Al2 . Atens21) . ImAl1 . DagAtens21]];
w*(Ggeo11*CothSafe[hbar*om001*om01*w/(2*kB*temp1)]
+Ggeo12*CothSafe[hbar*om001*om01*w/(2*kB*temp2)]
-Ggeo21*CothSafe[hbar*om001*om02*w/(2*kB*temp2)]
-Ggeo22*CothSafe[hbar*om001*om02*w/(2*kB*temp1)])
];


(* ================================================================*)
(*SECTION 10:Core integrand for nonlinear and nonlinear heat transfer*)
(*Note that this is only for radiation from particle 1 to particle 2*)
(*The exchange power should be the substraction between integLinear1 and integLinear2*)
(* ================================================================*)

(*Generic contraction:ContractTerm[Tleft,Btensor,ppp,Aright]; 
Tleft:3x3 matrix (e.g.T21^{-1});
Btensor:3x3x3 tensor (j,k,l indices) ;
ppp:3x3x3 tensor (k,l,m indices\[LongDash]the triple correlator);
Aright:3x3 matrix (conjugate,m,n indices) ;
Output:3x3 matrix C_{i,n}*)

(* ---------------------------------------------------------------------------- *)
(*  [V40] ContractTermLeft -- FLATTENED.  Identical to the ver39 Table/Sum form. *)
(*                                                                              *)
(*  Q12 notes Eq. (51):                                                         *)
(*      C_in = Sum_{j,k,l,m} [T21^-1]_ij [B21]_jkl <p_k(w1) p_l(w2) p+_m(w)>    *)
(*                            [A11^+]_mn                                        *)
(*                                                                              *)
(*  The notes call this a DOUBLE contraction: the pair (k,l) is contracted as a *)
(*  unit between B's two input-frequency legs and ppp's two source legs.        *)
(*  Grouping (k,l) into one composite index of dimension 9 turns that pair      *)
(*  contraction into a single matrix product:                                   *)
(*                                                                              *)
(*      Flatten[Btensor,{{1},{2,3}}]  -> 3 x 9 ,  row j , col (k,l)             *)
(*      Flatten[ppp,    {{1,2},{3}}]  -> 9 x 3 ,  row (k,l) , col m             *)
(*      ConjugateTranspose[Aright]    -> [m,n] = Conjugate[Aright[[n,m]]]       *)
(*                                       = [A^+]_mn , as in Eq. (51)            *)
(*                                                                              *)
(*  Flatten with level specifications is pure reindexing, so the result is the  *)
(*  same number, not an approximation.  It replaces 9 x 81 = 729 evaluator-     *)
(*  level Sum terms (each with 4 Part extractions and a Conjugate) by three     *)
(*  packed-array Dot products.                                                  *)
(* ---------------------------------------------------------------------------- *)
ContractTermLeft[Tleft_,Btensor_,ppp_,Aright_]:=
  Tleft . Flatten[Btensor,{{1},{2,3}}] . Flatten[ppp,{{1,2},{3}}] .
    ConjugateTranspose[Aright];

(* ---------------------------------------------------------------------------- *)
(*  [V40] ContractTermRight -- FLATTENED.  Identical to the ver39 Table/Sum form.*)
(*                                                                              *)
(*      C_in = Sum_{j,k,l} [leftM]_ij ppp[j,k,l] Conjugate[Bright[n,k,l]]       *)
(*                                                                              *)
(*  Same idea, with the (k,l) pair grouped on BOTH ppp and Bright:              *)
(*                                                                              *)
(*      Flatten[ppp,   {{1},{2,3}}]  -> 3 x 9 , row j , col (k,l)               *)
(*      Flatten[Bright,{{1},{2,3}}]  -> 3 x 9 , row n , col (k,l)               *)
(*                                                                              *)
(*  NOTE ON THE LEG ORDER (do not "fix" this):  the caller passes               *)
(*  Dagppp = Transpose[ppp,{2,3,1}], whose two daggered legs are ordered        *)
(*  (w1,w2).  Eq. (50) writes the conjugate correlator as                       *)
(*  <p(w) p+(w2) p+(w1)>, i.e. legs (w2,w1).  The two agree: Mtens is built as  *)
(*  M[i,q,s] = beta[i,j,k] A(w1)[j,q] A(w2)[k,s], so [B]_{n,k,l} carries        *)
(*  k <-> w1 and l <-> w2, and Transpose[ppp,{2,3,1}] is exactly the conjugate  *)
(*  correlator relabelled to that same (w1,w2) order.  The contraction is       *)
(*  therefore frequency-matched leg by leg.  Verified against SI_ver4           *)
(*  Eq. (corr_simplified).                                                      *)
(* ---------------------------------------------------------------------------- *)
ContractTermRight[leftM_,ppp_,Bright_]:=
  leftM . (Flatten[ppp,{{1},{2,3}}] .
           ConjugateTranspose[Flatten[Bright,{{1},{2,3}}]]);

(*TrIm of a 3x3 matrix*)
TrIm[M_]:=Im[Tr[M]];

(* ================================================================*)
(*  [F4] TRIPLE CORRELATOR  (SI-ver2_7 Eq. 65, symmetrized)          *)
(*                                                                  *)
(*  The absorbed power is the Hermitian operator (p.E + E.p)/2, so   *)
(*  the correlator entering <p E^dagger> is the symmetrized one,     *)
(*  (1/4)<{p^lam*(w),{p^mu(w1),p^nu(w2)}}>. Summing the four ordered *)
(*  correlators of Eq. (65) and using coth(a+b)=(1+CaCb)/(Ca+Cb)      *)
(*  gives the closed form                                            *)
(*                                                                  *)
(*   (1/4)<{p^lam*(w),{p^mu(w1),p^nu(w2)}}> = 2 Pi hbar^2 C x        *)
(*     [ C2 Re b^{mu nu lam}(w2,-w) + C1 Re b^{nu mu lam}(w1,-w)     *)
(*       - (C1+C2) Re b^{lam mu nu}(w1,w2) ] delta(w-w1-w2)          *)
(*                                                                  *)
(*  C = coth(hbar w/2kT), C1 = coth(hbar w1/2kT), C2 = coth(hbar w2/2kT).*)
(*  Same symmetrization as the coth linear FDT used in integL12.     *)
(*  Classical limit: 8 Pi (kT)^2 [X1/(w w2)+X2/(w w1)-X3/(w1 w2)],   *)
(*  i.e. SI-ver2_7 Eq. (66).                                         *)
(*                                                                  *)
(*  TripleKa returns the bracket times C (the factor 2 Pi is in      *)
(*  prefNL; hbar^2 is absorbed in the normalization as before).      *)
(*  Index order {mu (w1), nu (w2), lam (p^* at w)} -> ppp[[k,l,m]].  *)
(*  The B^dagger terms need the Hermitian conjugate, which for the    *)
(*  (real) symmetrized correlator is Transpose[ppp,{2,3,1}].          *)
(*  Mathematica Transpose[T,{n1,n2,n3}]: level k of T -> level nk.    *)
(* ================================================================*)

omOf[1]=om01; omOf[2]=om02;
cothT[j_,f_,T_]:=CothSafe[hbar*om001*omOf[j]*f/(2*kB*T)];

TripleKa[j_?IntegerQ,w_?NumericQ,w1_?NumericQ,T_?NumericQ]:=Module[{w2,C,C1,C2,X1,X2,X3,sB12,sB2N0,sB1N0},
  w2=w-w1;
  {sB12,sB2N0,sB1N0}=If[j==1,
     {s1B12[w,w1],s1B2N0[w,w1],s1B1N0[w,w1]},
     {s2B12[w,w1],s2B2N0[w,w1],s2B1N0[w,w1]}];
  C=cothT[j,w,T]; C1=cothT[j,w1,T]; C2=cothT[j,w2,T];
  X1=Re[sB2N0];                              (* Re b^{mu nu lam}(w2,-w)  *)
  X2=Transpose[Re[sB1N0],{2,1,3}];           (* Re b^{nu mu lam}(w1,-w)  *)
  X3=Transpose[Re[sB12],{3,1,2}];            (* Re b^{lam mu nu}(w1,w2)  *)
  C*(C2*X1+C1*X2-(C1+C2)*X3)];

integNL12[r2_?NumericQ,w_?NumericQ,w1_?NumericQ,temp1_?NumericQ,temp2_?NumericQ]:=Module[{w2,Rn,RRmat,II3,(*T matrices at w,w1,w2*)
T12w,T21w,T12w1,T21w1,T12w2,T21w2,(*Alpha tensors at w,w1,w2*)
Al1w,Al2w,Al1w1,Al2w1,Al1w2,Al2w2,(*Pinv at w,w1,w2*)
PinvMw,PinvMw1,PinvMw2,(*A tensors at w1 and w2 (needed for M tensors)*)
A11w1,A12w1,A21w1,A22w1,A11w2,A12w2,A21w2,A22w2,(*A tensors at w (needed for B tensors and integrand)*)
A11w,A12w,A21w,A22w,PinvNw,PinvNw1,PinvNw2,(*T inverses at w*)
T21invw,T12invw,(*beta scalars*)
beta1,beta2,(*M tensors*)Mtens111,Mtens112,Mtens221,Mtens222,(*B tensors*)
B11tens,B12tens,B21tens,B22tens,(*Triple correlators*)
ppp1,ppp2,Dagppp1,Dagppp2,prefNL,(*Terms*)
term11,term12,term13,term14,term21,term22,term23,term24, GgeoNL1,GgeoNL2},

w2=w-w1;
{Rn,RRmat}=geoData[r2];
II3=IdentityMatrix[3];

(*\[HorizontalLine]\[HorizontalLine] T matrices \[HorizontalLine]\[HorizontalLine]*)T12w=Tn12[w,Rn,RRmat];
T21w=Tn21[w,Rn,RRmat];
T12w1=Tn12[w1,Rn,RRmat];
T21w1=Tn21[w1,Rn,RRmat];
T12w2=Tn12[w2,Rn,RRmat];
T21w2=Tn21[w2,Rn,RRmat];

(*\[HorizontalLine]\[HorizontalLine] Alpha tensors \[HorizontalLine]\[HorizontalLine]*)
Al1w=alphaTens1[w];
Al2w=alphaTens2[w];
Al1w1=alphaTens1[w1];
Al2w1=alphaTens2[w1];
Al1w2=alphaTens1[w2];
Al2w2=alphaTens2[w2];

(*\[HorizontalLine]\[HorizontalLine] Pinv at each frequency \[HorizontalLine]\[HorizontalLine]*)
PinvMw=Inverse[II3-T12w . Al2w . T21w . Al1w];        (* P12^{-1} *)
PinvMw1=Inverse[II3-T12w1 . Al2w1 . T21w1 . Al1w1];
PinvMw2=Inverse[II3-T12w2 . Al2w2 . T21w2 . Al1w2];
PinvNw =PinvMw;    (* [F1 OFF] P21^{-1} -> P12^{-1} *)
PinvNw1=PinvMw1;
PinvNw2=PinvMw2;

(*\[HorizontalLine]\[HorizontalLine] A tensors at w (for B tensors and ContractTerm) \[HorizontalLine]\[HorizontalLine]*)
A11w=T12w . Al2w . PinvNw . T21w;   (* [F1] *)
A12w=PinvMw . T12w;
A21w=PinvNw . T21w;                (* [F1] *)
A22w=T21w . Al1w . PinvMw . T12w;
(* [F2] daggers are taken inside ContractTermLeft/Right; not needed here *)

(*\[HorizontalLine]\[HorizontalLine] A tensors at w1 (for M tensors) \[HorizontalLine]\[HorizontalLine]*)
A11w1=T12w1 . Al2w1 . PinvNw1 . T21w1;   (* [F1] *)
A12w1=PinvMw1 . T12w1;
A21w1=PinvNw1 . T21w1;                  (* [F1] *)
A22w1=T21w1 . Al1w1 . PinvMw1 . T12w1;

(*\[HorizontalLine]\[HorizontalLine] A tensors at w2 (for M tensors) \[HorizontalLine]\[HorizontalLine]*)
A11w2=T12w2 . Al2w2 . PinvNw2 . T21w2;   (* [F1] *)
A12w2=PinvMw2 . T12w2;
A21w2=PinvNw2 . T21w2;                  (* [F1] *)
A22w2=T21w2 . Al1w2 . PinvMw2 . T12w2;

(*\[HorizontalLine]\[HorizontalLine] T inverses at w \[HorizontalLine]\[HorizontalLine]*)
T21invw=Inverse[T21w];
T12invw=Inverse[T12w];

(*\[HorizontalLine]\[HorizontalLine] Beta scalars \[HorizontalLine]\[HorizontalLine]*)
beta1=s1B12[w,w1];
beta2=s2B12[w,w1];

(*\[HorizontalLine]\[HorizontalLine] M tensors:beta contracted with A at w1 and w2 \[HorizontalLine]\[HorizontalLine]*)
Mtens111=TensorContract[TensorProduct[beta1,A11w1,A11w2],{{2,4},{3,6}}];
Mtens112=TensorContract[TensorProduct[beta1,A12w1,A12w2],{{2,4},{3,6}}];
Mtens221=TensorContract[TensorProduct[beta2,A21w1,A21w2],{{2,4},{3,6}}];
Mtens222=TensorContract[TensorProduct[beta2,A22w1,A22w2],{{2,4},{3,6}}];

(*\[HorizontalLine]\[HorizontalLine] B tensors (inline,no global B11/B12/B21/B22 calls) \[HorizontalLine]\[HorizontalLine]*)
B11tens=T12w . Al2w . PinvNw . (T21w . Al1w . T12w . Mtens221+T21w . Mtens111)+T12w . Mtens221;   (* [F1] notes Eq. 26 *)
B12tens=PinvMw . (T12w . Al2w . T21w . Mtens112+T12w . Mtens222);                                (* notes Eq. 27 *)
B21tens=PinvNw . (T21w . Al1w . T12w . Mtens221+T21w . Mtens111);                                (* [F1] notes Eq. 28 *)
B22tens=T21w . Al1w . PinvMw . (T12w . Al2w . T21w . Mtens112+T12w . Mtens222)+T21w . Mtens112;
(* [F2] no ConjugateTranspose here (ContractTermRight conjugates B itself) *)

(*\[HorizontalLine]\[HorizontalLine] Triple correlators \[HorizontalLine]\[HorizontalLine]*)
(* [F3][F4] particle 1 at T1, particle 2 at T2 *)
ppp1=TripleKa[1,w,w1,temp1];          (* index order {mu,nu,lam} *)
ppp2=TripleKa[2,w,w1,temp2];
Dagppp1=Transpose[ppp1,{2,3,1}];      (* index order {lam,mu,nu} *)
Dagppp2=Transpose[ppp2,{2,3,1}];

(*\[HorizontalLine]\[HorizontalLine] Terms for Q1->2 (particle 1 absorbing) \[HorizontalLine]\[HorizontalLine]*)
(* notes Eq. 43; [F2] pass A and B (not their daggers); [F3] correct temperatures *)
term11=TrIm@ContractTermLeft[T21invw,B21tens,ppp1,A11w];
term12=TrIm@ContractTermLeft[T21invw,B22tens,ppp2,A12w];
term13=TrIm@ContractTermRight[T21invw . A21w,Dagppp1,B11tens];
term14=TrIm@ContractTermRight[T21invw . A22w,Dagppp2,B12tens];

(*\[HorizontalLine]\[HorizontalLine] Terms for Q2->1 (particle 2 absorbing) \[HorizontalLine]\[HorizontalLine]*)
term21=TrIm@ContractTermLeft[T12invw,B12tens,ppp2,A22w];
term22=TrIm@ContractTermLeft[T12invw,B11tens,ppp1,A21w];
term23=TrIm@ContractTermRight[T12invw . A12w,Dagppp2,B22tens];
term24=TrIm@ContractTermRight[T12invw . A11w,Dagppp1,B21tens];

(*\[HorizontalLine]\[HorizontalLine] Net heat transfer Q12-Q21 \[HorizontalLine]\[HorizontalLine]*)
(* [F4] 2 Pi (correlator)  x  1/(2 Pi) (d w/2 Pi)  x  1/(2 Pi) (sum over w1 = Integral d w1/2 Pi) *)
prefNL=(2*Pi)*1/(2*Pi)*1/(2*Pi);
GgeoNL2=prefNL*om02^3*(term21+term22+term23+term24);
GgeoNL1=prefNL*om01^3*(term11+term12+term13+term14);
w*(GgeoNL1-GgeoNL2)];


(* ================================================================*)
(*SECTION 11:The integral for the linear terms and the Compactified double integral for the nonlinear terms*)
(* ================================================================*)
(**)
(*Change of variables:*)
(*u=w/(1+w) in[0,1],w=u/(1-u)*)
(*v=w1/w in[0,1],w1=v*w*)
(**)
(*Jacobian:dw/du=1/(1-u)^2*)
(*dw1/dv=w (for fixed u)*)
(*            =>d(w,w1)/d(u,v)=w/(1-u)^2*)
(**)
(*Net power of w in integrand:*)
(*prefactor contributes 1/w*)
(*Jacobian contributes w*)
(*    =>integrand~w^0 at u=0 (finite,no singularity)*)
(**)
(*Boundary v=1 (w1=w,w2=0):handled by CothReg.*)
(*No Exclusions needed.*)
(* ================================================================*)

QL12[r2_?NumericQ]:=NIntegrate[integL12[r2,w,T1,T2],{w,0,Infinity},Method->{"GlobalAdaptive","SingularityHandler"->"IMT"},AccuracyGoal->6,PrecisionGoal->6,MaxRecursion->100];

QNL12[r2_?NumericQ]:=NIntegrate[
(u/(1-u)^3)*integNL12[r2,u/(1-u),u*v/(1-u),T1,T2],{u,0,1},{v,0,1},Method->{"GlobalAdaptive","SingularityHandler"->"IMT","MaxErrorIncreases"->100},AccuracyGoal->4,PrecisionGoal->4,MaxRecursion->60];

r2ListUnit=Join[Table[10^i,{i,Log10[1],1-1/50,1/50}],Table[10^i,{i,1,3,1/50}]];
r2List=r2ListUnit/EffR2;

(* ---- window of the Q-vs-R figure (needed by the NL map even if stage 1 is skipped) ---- *)
r2LoA = EffR2;      (* left edge of the r2 window, Angstrom  *)
r2HiA = 1000.0;     (* right edge of the r2 window, Angstrom *)

(* ============================================================================ *)
(*  STAGE 1 : Q vs R   (ver41: wrapped in If[doQvsR, ...]; QTotalList reuses    *)
(*            the two lists instead of recomputing QL12 and QNL12 again)        *)
(* ============================================================================ *)
If[doQvsR,
Print[DateString[], "  stage 1: Q vs R"];
QL12List=Table[Module[{val},
val=hbar*om001^2*QL12[r2];
{r2,val}],
{r2,r2List}];

QNL12List=Table[Module[{val},
val=hbar*om001^2*QNL12[r2];
{r2,val}],
{r2,r2List}];

(* [ver41] identical numbers to ver40's QTotalList, without recomputing both integrals *)
QTotalList = MapThread[{#1[[1]], #1[[2]] + #2[[2]]} &, {QL12List, QNL12List}];

QL12ListUnit={#[[1]]*EffR2,#[[2]]}&/@QL12List;
QNL12ListUnit={#[[1]]*EffR2,#[[2]]}&/@QNL12List;
QTotalListUnit={#[[1]]*EffR2,#[[2]]}&/@QTotalList;

xLo = Log10[r2LoA];  xHi = Log10[r2HiA];
With[{allY = Log10[Abs[#[[2]]]] & /@
       Select[Join[QNL12ListUnit, QL12ListUnit, QTotalListUnit],
              r2LoA <= #[[1]] <= r2HiA && #[[2]] != 0 &]},
   yLo = Floor[Min[allY]];  yHi = Ceiling[Max[allY]]];

(* ============================================================ *)
(*  DATA EXPORT FOR ORIGINLAB  --  Q vs R                       *)
(*  Six series, sign-split (pos/neg) in |Q|.                     *)
(* ============================================================ *)
Module[{clean, mkCol, series, maxLen, pad, tbl, hdr, meta},

  clean[x_] := Chop[Re[N[x]], 10^-30];

  (* same window + sign split + |Q| as the figure *)
  mkCol[data_, sgn_] := Module[{d},
     d = Select[data, r2LoA <= #[[1]] <= r2HiA &];
     d = Select[d, sgn*clean[Last[#]] > 0 &];
     {clean[#[[1]]], Abs[clean[#[[2]]]]} & /@ d];

  series = {
     {"QNL_pos",   mkCol[QNL12ListUnit,   1]},
     {"QNL_neg",   mkCol[QNL12ListUnit,  -1]},
     {"QL_pos",    mkCol[QL12ListUnit,    1]},
     {"QL_neg",    mkCol[QL12ListUnit,   -1]},
     {"QTot_pos",  mkCol[QTotalListUnit,  1]},
     {"QTot_neg",  mkCol[QTotalListUnit, -1]}};

  (* each series has its own R values -> pad to a common length *)
  maxLen = Max[Length /@ series[[All, 2]]];
  pad[d_] := Join[d, ConstantArray[{"", ""}, maxLen - Length[d]]];

  tbl = Transpose[Flatten[pad[#[[2]]], {{1}, {2}}] & /@ series, {2, 1, 3}];
  tbl = Flatten[#, 1] & /@ tbl;

  hdr = Flatten[{"R_" <> #[[1]], "absQ_" <> #[[1]]} & /@ series];

  (* force plain decimals: numericize any exact number, leave "" padding as-is *)
  tbl = Map[If[NumberQ[#] || Head[#] === Rational || Head[#] === Integer,
               N[#], #] &, tbl, {2}];
  Export[FileNameJoin[{outDir, "Q_vs_R_series.csv"}], Prepend[tbl, hdr], "CSV"];

  (* the axis limits the figure computed, so Origin can match them *)
  meta = {{"xLo_log10", N[xLo]}, {"xHi_log10", N[xHi]},
          {"yLo_log10", N[yLo]}, {"yHi_log10", N[yHi]},
          {"R_min_Ang", N[10.^xLo]}, {"R_max_Ang", N[10.^xHi]},
          {"absQ_min",  N[10.^yLo]}, {"absQ_max",  N[10.^yHi]},
          {"EffR2_Ang", N[EffR2]}};
  Export[FileNameJoin[{outDir, "Q_vs_R_axes.csv"}],
     Prepend[meta, {"quantity", "value"}], "CSV"];

  Print["Q vs R export -> ", outDir];
  Print["  series rows: ", maxLen, "   counts: ",
        {#[[1]], Length[#[[2]]]} & /@ series];
];
Print[DateString[], "  stage 1 done"],
Print["stage 1 (Q vs R) skipped  (runQvsR = ", runQvsR, ", NL_MODE = ", nlMode, "; Automatic skips it when Q_vs_R_series.csv exists)"]];

(* ============================================================================ *)
(* ============================================================================ *)
(*                                                                     *)
(*   CELL A  -  HEAT MAP OF  integL12                                           *)
(*             x = separation r2 (Angstrom) , y = hbar*omega (eV)               *)
(*                                                                     *)
(* ============================================================================ *)
(* ============================================================================ *)

(* ---- window : same R range as Figure Q-vs-R; frequency range shared with integNL12 map ---- *)
r2MinA = r2LoA;  r2MaxA = r2HiA;   nR2 = 150;   (* same R window as Figure Q-vs-R *)
eMinEV = 0.001; eMaxEV = 10.0;     nW  = 150;   (* photon energy, eV, log-spaced    *)
cFloorFrac  = 1.*^-22;                          (* signed-log floor (same as NL map)*)
useParallel = True;
plotImageSize = 540;

(* ---- to LOCK this map to the nonlinear map's exact color scale, paste its     *)
(*      cFloor / zmax here; leave Automatic for independent (same-style) scaling. *)
cFloorShared = cFloorGlobal;   (* absolute floor, shared with integNL12 (COLORBAR panel) *)
zmaxShared   = zmaxGlobal;     (* absolute color-axis limit, shared with integNL12        *)

(* ---- grids ---- *)
r2AngGrid = 10^Range[Log10[r2MinA], Log10[r2MaxA], (Log10[r2MaxA] - Log10[r2MinA])/(nR2 - 1)];
eGrid     = 10^Range[Log10[eMinEV], Log10[eMaxEV], (Log10[eMaxEV] - Log10[eMinEV])/(nW - 1)];

If[doL12Map,
Print[DateString[], "  stage 2: integL12 map"];
ensureKernels[];                                    (* [ver41] was LaunchKernels[] *)
If[useParallel, DistributeDefinitions["Global`"]];
tbl = If[useParallel, ParallelTable, Table];
rawL = Flatten[tbl[
    {en, rA, Quiet@Check[integL12[rA/EffR2, en/(hbar*om001), T1, T2], Indeterminate]},
    {rA, r2AngGrid}, {en, eGrid}], 1];
rawL = Select[rawL, NumericQ[#[[3]]] &];

(* ---- SIGNED-LOG floor : shared & absolute (from COLORBAR CONTROL PANEL) ---- *)
vAbsMax = Max[Abs[rawL[[All, 3]]]];
cFloor  = If[cFloorShared === Automatic, vAbsMax*cFloorFrac, cFloorShared];  (* = cFloorGlobal *)
zmax    = If[zmaxShared   === Automatic,
             Max[Abs[Sign[#]*Log10[1 + Abs[#]/cFloor] & /@ rawL[[All, 3]]]],
             zmaxShared];                                                    (* = zmaxGlobal   *)

Print["integL12  raw min/max : ", ScientificForm[MinMax[rawL[[All, 3]]], 3]];
Print["integL12  cFloor = ", ScientificForm[cFloor, 3],
      "   color range = [", NumberForm[-zmax, 4], ", ", NumberForm[zmax, 4], "]"];

(* ---- colorbar metadata : lets Origin lock BOTH maps to one identical scale ---- *)
Export[FileNameJoin[{outDir, "integL12_colorbar.csv"}],
   {{"quantity", "value"}, {"cFloor", cFloor},
    {"color_lo", -zmax}, {"color_hi", zmax}}, "CSV"];


(* ============================================================================ *)
(*  DATA EXPORT FOR ORIGINLAB  --  integL12 heat map                            *)
(*  Reuses the arrays already computed above (no recomputation).                *)
(*  Writes into the same ./figures directory as the PDF/PNG figures.            *)
(*                                                                              *)
(*    integL12_XYZ.csv               long/XYZ : R(Ang), E(eV), raw, signedlog   *)
(*    integL12_matrix_signedlog.csv  grid : X=log10(R) in row 1, Y=log10(E)     *)
(*                                    in col 1, cell = sgn*log10(1+|v|/cFloor)   *)
(*    integL12_matrix_raw.csv        same grid, cell = raw integL12             *)
(*                                                                              *)
(*  In Origin: color-map the 'signedlog' column/matrix with a diverging        *)
(*  (blue-white-red) palette centered at 0 to reproduce the Mathematica look.  *)
(* ============================================================================ *)
Module[{slOf, xyz, xs, ys, lookup, mRaw, mSL},
  slOf[v_] := Sign[v]*Log10[1 + Abs[v]/cFloor];   (* cFloor here = integL12 floor *)

  (* (1) LONG / XYZ  -- rawL = {E(eV), R(Ang), value}; reorder to R,E,raw,SL *)
  xyz = {#[[2]], #[[1]], #[[3]], slOf[#[[3]]]} & /@ rawL;
  xyz = SortBy[xyz, {First, #[[2]] &}];
  Export[FileNameJoin[{outDir, "integL12_XYZ.csv"}],
     Prepend[xyz, {"R_Angstrom", "E_eV", "integL12", "signedlog"}], "CSV"];

  (* (2) MATRIX  -- grid is regular in log-log; index by log10(R) x log10(E) *)
  xs = Log10[r2AngGrid];        (* column coords (evenly spaced)             *)
  ys = Log10[eGrid];            (* row    coords (evenly spaced)             *)
  lookup = Association[({#[[2]], #[[1]]} -> #[[3]]) & /@ rawL];   (* {R,E}->v *)
  mRaw = Table[Lookup[lookup, Key[{r, e}], ""], {e, eGrid}, {r, r2AngGrid}];
  mSL  = Map[If[NumericQ[#], slOf[#], ""] &, mRaw, {2}];
  Export[FileNameJoin[{outDir, "integL12_matrix_signedlog.csv"}],
     Prepend[MapThread[Prepend, {mSL, ys}], Prepend[xs, "logE_logR"]], "CSV"];
  Export[FileNameJoin[{outDir, "integL12_matrix_raw.csv"}],
     Prepend[MapThread[Prepend, {mRaw, ys}], Prepend[xs, "logE_logR"]], "CSV"];

  Print["OriginLab export (integL12) -> ", outDir];
  Print["  integL12_XYZ.csv , integL12_matrix_signedlog.csv , integL12_matrix_raw.csv"];
  Print["  signed-log floor cFloor = ", ScientificForm[cFloor, 3]];
];
Print[DateString[], "  stage 2 done"],
Print["stage 2 (integL12 map) skipped  (runL12Map = ", runL12Map, ", NL_MODE = ", nlMode, "; Automatic skips it when integL12_XYZ.csv exists)"]];

(* ============================================================================ *)
(*   HEAT MAP OF  integNL12   -   ver41: thermal-only integrand, checkpointed   *)
(*                                                                              *)
(*   See the ver41 header for the reasons.  In short:                           *)
(*     integNL12Th  = integNL12 with the zero-point part of the triple          *)
(*                    correlator removed (exact for identical dipoles; checked  *)
(*                    by nlSelfTest before use)                                 *)
(*     one option set, PrecisionGoal 6, breakpoints at thermal scales           *)
(*     rows written to figures_noF1/NL_rows/row_###.m as they finish            *)
(* ============================================================================ *)

(* (S1) keep the geoData memo (only ~105 distinct r2 values).  The alphaTens1/2  *)
(*      memo of ver40 is REMOVED: it is keyed on w1, w2 which never repeat.      *)
geoData[r2_?NumericQ]   := geoData[r2] = Module[{Rvec, Rn, Rhat},
   Rvec = r1n - r2n[r2]; Rn = Norm[Rvec]; Rhat = Rvec/Rn;
   {Rn, Outer[Times, Rhat, Rhat]}];

(* ############################################################################ *)
(* #####                       CONTROL PANEL                             ###### *)
(* ############################################################################ *)

temp1 = T1;  temp2 = T2;

(* ---- window (unchanged from ver40) ---- *)
r2MinAng = EffR2;  r2MaxAng = 200.0;
wMinEV   = 0.01;   wMaxEV   = 10.0;

(* ---- hold the mesh density of the ORIGINAL window (unchanged) ---- *)
densR2 = 150/(Log10[r2HiA]  - Log10[r2LoA]);    (* points per decade in R *)
densW  = 150/(Log10[eMaxEV] - Log10[eMinEV]);   (* points per decade in E *)

nR2 = Round[densR2*(Log10[r2MaxAng] - Log10[r2MinAng])];
nW  = Round[densW *(Log10[wMaxEV]   - Log10[wMinEV])];

(* ---- resonance grid refinement (unchanged) ---- *)
refineResonance = True;
nWidths = 60;
nResPts = 403;

(* ---- [ver41] integration control ------------------------------------------ *)
nlThermalOnly = Automatic;  (* Automatic: use integNL12Th if the self-test passes *)
nlHalfDomain  = Automatic;  (* Automatic: use v in (0,1/2) x 2 if f(v)=f(1-v)     *)
nlPrec        = 6;          (* PrecisionGoal (relative) for every row             *)
nlAcc         = Infinity;   (* AccuracyGoal: none needed once the noise is gone.  *)
                            (* e.g. 24 would also stop at |err| < 1e-24, i.e.     *)
                            (* 100x below cFloorGlobal - invisible in the map.    *)
nlMaxRec      = 50;
nlMaxErrInc   = 100;
nlThermalBP   = {1, 4, 16, 64};  (* breakpoints at m*kT/(hbar w0 w), T = T1, T2  *)
bpHalfWidths  = 6;               (* resonance nodes at w0/w +- 6 eta/w (as ver40) *)
skipSingularRows = True;         (* skip rows with det(I - T a T a)(w->0) ~ 0     *)
keepFlagged   = False;           (* False = drop warned points from maps (ver40)  *)

(* ---- color: signed-log (SAME colorbar & pipeline as the integL12 map) ------- *)
cFloorNL = cFloorGlobal;

r2ToAng[r2_] := r2*EffR2;
EofW[w_]     := hbar*om001*w;

r2Min0 = N[r2MinAng/EffR2];       r2Max0 = N[r2MaxAng/EffR2];
wMin0  = N[wMinEV/(hbar*om001)];  wMax0  = N[wMaxEV/(hbar*om001)];

wRes   = N[om002/om001];
etaRel = N[eta1/om001];
Print["resonance at w = ", wRes, "  (hbar*w = ", NumberForm[EofW[wRes], 3], " eV)"];
Print["relative linewidth etaRel = ", ScientificForm[etaRel, 3],
      "   (", NumberForm[EofW[etaRel], 3], " eV)"];

(* ---------------------------------------------------------------------------- *)
(*  GRIDS  (identical to ver40)                                                  *)
(* ---------------------------------------------------------------------------- *)
wgridLog = N[10^Subdivide[Log10[wMin0], Log10[wMax0], nW - 1]];
wgridRes = If[TrueQ[refineResonance],
   Select[N@Subdivide[wRes - nWidths*etaRel, wRes + nWidths*etaRel, nResPts - 1],
          wMin0 <= # <= wMax0 &], {}];
wgrid  = DeleteDuplicates[Sort[N@Join[wgridLog, wgridRes]], Abs[#1 - #2] < 10^-12 &];
r2grid = N[10^Subdivide[Log10[r2Min0], Log10[r2Max0], nR2 - 1]];

Print["w-grid: ", Length[wgridLog], " log + ", Length[wgridRes],
      " resonance = ", Length[wgrid], " rows;   r2-grid: ", Length[r2grid], " rows"];

(* ---------------------------------------------------------------------------- *)
(*  [G4] rows on the static polarization catastrophe                             *)
(*       det(I - T12 a2 T21 a1) -> 0 as w -> 0 : the v-integral diverges there. *)
(* ---------------------------------------------------------------------------- *)
staticDetAbs[r2_?NumericQ] := Module[{Rn, RR, x = 1.*^-6},
   {Rn, RR} = geoData[r2];
   Abs[Det[IdentityMatrix[3] - Tn12[x, Rn, RR] . alphaTens2[x] . Tn21[x, Rn, RR] . alphaTens1[x]]]];
singularRows = If[TrueQ[skipSingularRows],
   Select[Range[Length[r2grid]], staticDetAbs[r2grid[[#]]] < 1.*^-4 &], {}];
Print["rows skipped (polarization catastrophe at w->0): ",
      {#, NumberForm[r2ToAng[r2grid[[#]]], 5]} & /@ singularRows];
activeRows = Complement[Range[Length[r2grid]], singularRows];

(* ---------------------------------------------------------------------------- *)
(*  [G1] THERMAL-ONLY TRIPLE CORRELATOR                                          *)
(*    coth(x/2) = 1 + 2 n(x),  n(x) = 1/(e^x - 1),  x = hbar w/(kB T)            *)
(*    C C2 - 1       = 2n + 2n2 + 4 n n2                                         *)
(*    C C1 - 1       = 2n + 2n1 + 4 n n1                                         *)
(*    C (C1+C2) - 2  = 2n1 + 2n2 + 4n + 4n (n1+n2)                               *)
(*    so  TripleKa = TripleKa0 + TripleKaTh  exactly, with                       *)
(*    TripleKa0 = X1 + X2 - 2 X3  (the T-independent zero-point part).           *)
(* ---------------------------------------------------------------------------- *)
nOccT[j_, f_?NumericQ, T_?NumericQ] := Module[{y},
   y = hbar*om001*omOf[j]*f/(kB*T);
   Which[y > 700., 0., y < 1.*^-4, 1/y - 0.5 + y/12, True, 1/(Exp[y] - 1)]];

TripleKaTh[j_?IntegerQ, w_?NumericQ, w1_?NumericQ, T_?NumericQ] := Module[
   {w2, n0, n1, n2, X1, X2, X3, sB12, sB2N0, sB1N0},
   w2 = w - w1;
   {sB12, sB2N0, sB1N0} = If[j == 1,
      {s1B12[w, w1], s1B2N0[w, w1], s1B1N0[w, w1]},
      {s2B12[w, w1], s2B2N0[w, w1], s2B1N0[w, w1]}];
   n0 = nOccT[j, w, T]; n1 = nOccT[j, w1, T]; n2 = nOccT[j, w2, T];
   X1 = Re[sB2N0];
   X2 = Transpose[Re[sB1N0], {2, 1, 3}];
   X3 = Transpose[Re[sB12], {3, 1, 2}];
   (2 n0 + 2 n2 + 4 n0 n2)*X1 + (2 n0 + 2 n1 + 4 n0 n1)*X2 -
      (2 n1 + 2 n2 + 4 n0 + 4 n0 (n1 + n2))*X3];

TripleKa0[j_?IntegerQ, w_?NumericQ, w1_?NumericQ, T_?NumericQ] := Module[
   {X1, X2, X3, sB12, sB2N0, sB1N0},
   {sB12, sB2N0, sB1N0} = If[j == 1,
      {s1B12[w, w1], s1B2N0[w, w1], s1B1N0[w, w1]},
      {s2B12[w, w1], s2B2N0[w, w1], s2B1N0[w, w1]}];
   X1 = Re[sB2N0];
   X2 = Transpose[Re[sB1N0], {2, 1, 3}];
   X3 = Transpose[Re[sB12], {3, 1, 2}];
   X1 + X2 - 2*X3];

(* integNL12parts: the body of integNL12 copied VERBATIM, except that the triple  *)
(* correlator function is an argument and it returns {w GgeoNL1, w GgeoNL2}.     *)
(* integNL12 itself is untouched and is used by the self-test as the reference.  *)
integNL12parts[r2_?NumericQ,w_?NumericQ,w1_?NumericQ,temp1_?NumericQ,temp2_?NumericQ,pppFun_]:=Module[{w2,Rn,RRmat,II3,(*T matrices at w,w1,w2*)
T12w,T21w,T12w1,T21w1,T12w2,T21w2,(*Alpha tensors at w,w1,w2*)
Al1w,Al2w,Al1w1,Al2w1,Al1w2,Al2w2,(*Pinv at w,w1,w2*)
PinvMw,PinvMw1,PinvMw2,(*A tensors at w1 and w2 (needed for M tensors)*)
A11w1,A12w1,A21w1,A22w1,A11w2,A12w2,A21w2,A22w2,(*A tensors at w (needed for B tensors and integrand)*)
A11w,A12w,A21w,A22w,PinvNw,PinvNw1,PinvNw2,(*T inverses at w*)
T21invw,T12invw,(*beta scalars*)
beta1,beta2,(*M tensors*)Mtens111,Mtens112,Mtens221,Mtens222,(*B tensors*)
B11tens,B12tens,B21tens,B22tens,(*Triple correlators*)
ppp1,ppp2,Dagppp1,Dagppp2,prefNL,(*Terms*)
term11,term12,term13,term14,term21,term22,term23,term24, GgeoNL1,GgeoNL2},

w2=w-w1;
{Rn,RRmat}=geoData[r2];
II3=IdentityMatrix[3];

(*\[HorizontalLine]\[HorizontalLine] T matrices \[HorizontalLine]\[HorizontalLine]*)T12w=Tn12[w,Rn,RRmat];
T21w=Tn21[w,Rn,RRmat];
T12w1=Tn12[w1,Rn,RRmat];
T21w1=Tn21[w1,Rn,RRmat];
T12w2=Tn12[w2,Rn,RRmat];
T21w2=Tn21[w2,Rn,RRmat];

(*\[HorizontalLine]\[HorizontalLine] Alpha tensors \[HorizontalLine]\[HorizontalLine]*)
Al1w=alphaTens1[w];
Al2w=alphaTens2[w];
Al1w1=alphaTens1[w1];
Al2w1=alphaTens2[w1];
Al1w2=alphaTens1[w2];
Al2w2=alphaTens2[w2];

(*\[HorizontalLine]\[HorizontalLine] Pinv at each frequency \[HorizontalLine]\[HorizontalLine]*)
PinvMw=Inverse[II3-T12w . Al2w . T21w . Al1w];        (* P12^{-1} *)
PinvMw1=Inverse[II3-T12w1 . Al2w1 . T21w1 . Al1w1];
PinvMw2=Inverse[II3-T12w2 . Al2w2 . T21w2 . Al1w2];
PinvNw =PinvMw;    (* [F1 OFF] P21^{-1} -> P12^{-1} *)
PinvNw1=PinvMw1;
PinvNw2=PinvMw2;

(*\[HorizontalLine]\[HorizontalLine] A tensors at w (for B tensors and ContractTerm) \[HorizontalLine]\[HorizontalLine]*)
A11w=T12w . Al2w . PinvNw . T21w;   (* [F1] *)
A12w=PinvMw . T12w;
A21w=PinvNw . T21w;                (* [F1] *)
A22w=T21w . Al1w . PinvMw . T12w;
(* [F2] daggers are taken inside ContractTermLeft/Right; not needed here *)

(*\[HorizontalLine]\[HorizontalLine] A tensors at w1 (for M tensors) \[HorizontalLine]\[HorizontalLine]*)
A11w1=T12w1 . Al2w1 . PinvNw1 . T21w1;   (* [F1] *)
A12w1=PinvMw1 . T12w1;
A21w1=PinvNw1 . T21w1;                  (* [F1] *)
A22w1=T21w1 . Al1w1 . PinvMw1 . T12w1;

(*\[HorizontalLine]\[HorizontalLine] A tensors at w2 (for M tensors) \[HorizontalLine]\[HorizontalLine]*)
A11w2=T12w2 . Al2w2 . PinvNw2 . T21w2;   (* [F1] *)
A12w2=PinvMw2 . T12w2;
A21w2=PinvNw2 . T21w2;                  (* [F1] *)
A22w2=T21w2 . Al1w2 . PinvMw2 . T12w2;

(*\[HorizontalLine]\[HorizontalLine] T inverses at w \[HorizontalLine]\[HorizontalLine]*)
T21invw=Inverse[T21w];
T12invw=Inverse[T12w];

(*\[HorizontalLine]\[HorizontalLine] Beta scalars \[HorizontalLine]\[HorizontalLine]*)
beta1=s1B12[w,w1];
beta2=s2B12[w,w1];

(*\[HorizontalLine]\[HorizontalLine] M tensors:beta contracted with A at w1 and w2 \[HorizontalLine]\[HorizontalLine]*)
Mtens111=TensorContract[TensorProduct[beta1,A11w1,A11w2],{{2,4},{3,6}}];
Mtens112=TensorContract[TensorProduct[beta1,A12w1,A12w2],{{2,4},{3,6}}];
Mtens221=TensorContract[TensorProduct[beta2,A21w1,A21w2],{{2,4},{3,6}}];
Mtens222=TensorContract[TensorProduct[beta2,A22w1,A22w2],{{2,4},{3,6}}];

(*\[HorizontalLine]\[HorizontalLine] B tensors (inline,no global B11/B12/B21/B22 calls) \[HorizontalLine]\[HorizontalLine]*)
B11tens=T12w . Al2w . PinvNw . (T21w . Al1w . T12w . Mtens221+T21w . Mtens111)+T12w . Mtens221;   (* [F1] notes Eq. 26 *)
B12tens=PinvMw . (T12w . Al2w . T21w . Mtens112+T12w . Mtens222);                                (* notes Eq. 27 *)
B21tens=PinvNw . (T21w . Al1w . T12w . Mtens221+T21w . Mtens111);                                (* [F1] notes Eq. 28 *)
B22tens=T21w . Al1w . PinvMw . (T12w . Al2w . T21w . Mtens112+T12w . Mtens222)+T21w . Mtens112;
(* [F2] no ConjugateTranspose here (ContractTermRight conjugates B itself) *)

(*\[HorizontalLine]\[HorizontalLine] Triple correlators \[HorizontalLine]\[HorizontalLine]*)
(* [F3][F4] particle 1 at T1, particle 2 at T2 *)
ppp1=pppFun[1,w,w1,temp1];          (* index order {mu,nu,lam} *)
ppp2=pppFun[2,w,w1,temp2];
Dagppp1=Transpose[ppp1,{2,3,1}];      (* index order {lam,mu,nu} *)
Dagppp2=Transpose[ppp2,{2,3,1}];

(*\[HorizontalLine]\[HorizontalLine] Terms for Q1->2 (particle 1 absorbing) \[HorizontalLine]\[HorizontalLine]*)
(* notes Eq. 43; [F2] pass A and B (not their daggers); [F3] correct temperatures *)
term11=TrIm@ContractTermLeft[T21invw,B21tens,ppp1,A11w];
term12=TrIm@ContractTermLeft[T21invw,B22tens,ppp2,A12w];
term13=TrIm@ContractTermRight[T21invw . A21w,Dagppp1,B11tens];
term14=TrIm@ContractTermRight[T21invw . A22w,Dagppp2,B12tens];

(*\[HorizontalLine]\[HorizontalLine] Terms for Q2->1 (particle 2 absorbing) \[HorizontalLine]\[HorizontalLine]*)
term21=TrIm@ContractTermLeft[T12invw,B12tens,ppp2,A22w];
term22=TrIm@ContractTermLeft[T12invw,B11tens,ppp1,A21w];
term23=TrIm@ContractTermRight[T12invw . A12w,Dagppp2,B22tens];
term24=TrIm@ContractTermRight[T12invw . A11w,Dagppp1,B21tens];

(*\[HorizontalLine]\[HorizontalLine] Net heat transfer Q12-Q21 \[HorizontalLine]\[HorizontalLine]*)
(* [F4] 2 Pi (correlator)  x  1/(2 Pi) (d w/2 Pi)  x  1/(2 Pi) (sum over w1 = Integral d w1/2 Pi) *)
prefNL=(2*Pi)*1/(2*Pi)*1/(2*Pi);
GgeoNL2=prefNL*om02^3*(term21+term22+term23+term24);
GgeoNL1=prefNL*om01^3*(term11+term12+term13+term14);
{w*GgeoNL1,w*GgeoNL2}];

integNL12Th[r2_?NumericQ, w_?NumericQ, w1_?NumericQ, t1_?NumericQ, t2_?NumericQ] :=
   Subtract @@ integNL12parts[r2, w, w1, t1, t2, TripleKaTh];

(* ---------------------------------------------------------------------------- *)
(*  SELF-TEST  (runs in seconds; decides nlThermalOnly / nlHalfDomain)           *)
(* ---------------------------------------------------------------------------- *)
nlSelfTest[] := Module[{rowsT, wsT, vsT, pts, rows, corr, t0, nEv = 12},
   rowsT = activeRows[[Union[Round[Rescale[{0., 0.12, 0.4, 0.7, 1.}, {0., 1.}, {1, Length[activeRows]}]]]]];
   wsT = {0.3, 0.9, 1.0, 1.2, 2.0};
   vsT = {0.003, 0.05, 0.3};
   pts = Tuples[{r2grid[[rowsT]], wsT, vsT}];
   (* (a) correlator identity  TripleKa = TripleKa0 + TripleKaTh  (catches formula/index errors) *)
   corr = AllTrue[Tuples[{{1, 2}, wsT, vsT}], Function[q,
      Module[{j = q[[1]], w = q[[2]], v = q[[3]], tt, pF, p0, pT},
         tt = If[j == 1, temp1, temp2];
         pF = TripleKa[j, w, v*w, tt];  p0 = TripleKa0[j, w, v*w, tt];  pT = TripleKaTh[j, w, v*w, tt];
         Max[Abs[pF - p0 - pT]] <= 1.*^-7 Max[Abs[pF], Abs[p0], 1.*^-300]]]];
   (* (b) full integrand vs pieces.  'sig' = thermal window (v <= 0.05), where the net   *)
   (*     is far above the zero-point round-off of the ORIGINAL form; at v = 0.3 the     *)
   (*     original form is mostly noise, so only the exact checks are applied there.     *)
   rows = Map[Function[p, Module[{r2 = p[[1]], w = p[[2]], v = p[[3]], full, pf, pt, p0, th, z, sc, thS, fullS, sig},
         full  = integNL12[r2, w, v*w, temp1, temp2];
         pf    = integNL12parts[r2, w, v*w, temp1, temp2, TripleKa];
         pt    = integNL12parts[r2, w, v*w, temp1, temp2, TripleKaTh];
         p0    = integNL12parts[r2, w, v*w, temp1, temp2, TripleKa0];
         th    = pt[[1]] - pt[[2]];
         z     = p0[[1]] - p0[[2]];
         sc    = Max[Abs[p0], Abs[pf], 1.*^-300];
         thS   = integNL12Th[r2, w, w - v*w, temp1, temp2];
         fullS = integNL12[r2, w, w - v*w, temp1, temp2];
         sig   = v <= 0.05;
         <|"copy"    -> Abs[full - (pf[[1]] - pf[[2]])] <= 1.*^-10 Abs[full] + 1.*^-14 sc,
           "zero"    -> Abs[z] <= 1.*^-9 sc,
           "agree"   -> (!sig || Abs[full - th] <= 1.*^-6 Abs[th] + 1.*^-10 sc),
           "symTh"   -> Abs[th - thS] <= 1.*^-6 Abs[th] + 1.*^-12 sc,
           "symFull" -> (!sig || Abs[full - fullS] <= 1.*^-6 Abs[full] + 1.*^-10 sc),
           "relDiff" -> If[sig, Abs[full - th]/Max[Abs[th], 1.*^-300], 0.]|>]], pts];
   t0 = First@AbsoluteTiming[Do[integNL12Th[r2grid[[rowsT[[Min[2, Length[rowsT]]]]]], 1.2, 0.01*k, temp1, temp2], {k, nEv}]]/nEv;
   <|"copy" -> AllTrue[rows, TrueQ[#["copy"]] &], "corr" -> corr, "zero" -> AllTrue[rows, TrueQ[#["zero"]] &],
     "agree" -> AllTrue[rows, TrueQ[#["agree"]] &], "symTh" -> AllTrue[rows, TrueQ[#["symTh"]] &],
     "symFull" -> AllTrue[rows, TrueQ[#["symFull"]] &],
     "maxRelDiff" -> Max[#["relDiff"] & /@ rows], "nPts" -> Length[pts], "tEval" -> t0|>];

Print[DateString[], "  running ver41 self-test ..."];
nlTest = nlSelfTest[];
Print["  self-test on ", nlTest["nPts"], " points:"];
Print["    integNL12parts reproduces integNL12 : ", nlTest["copy"]];
Print["    TripleKa = TripleKa0 + TripleKaTh    : ", nlTest["corr"]];
Print["    zero-point part of the net = 0       : ", nlTest["zero"], "   (identical dipoles)"];
Print["    thermal-only == original (v<=0.05)   : ", nlTest["agree"],
      "   max rel diff ", ScientificForm[nlTest["maxRelDiff"], 2]];
Print["    f(v) = f(1-v)  thermal / original    : ", nlTest["symTh"], " / ", nlTest["symFull"]];
Print["  one integrand evaluation (master kernel): ", NumberForm[1000.*nlTest["tEval"], 3], " ms"];

useThermal = Switch[nlThermalOnly, True, True, False, False, _,
   TrueQ[nlTest["copy"] && nlTest["corr"] && nlTest["zero"] && nlTest["agree"]]];
If[!TrueQ[nlTest["copy"]],
   Print["*** WARNING: integNL12parts does not reproduce integNL12 -> falling back to integNL12"];
   useThermal = False];
If[TrueQ[nlThermalOnly] && !TrueQ[nlTest["zero"] && nlTest["corr"] && nlTest["agree"]],
   Print["*** WARNING: nlThermalOnly = True was forced although the self-test failed."]];
useHalf = Switch[nlHalfDomain, True, True, False, False, _,
   TrueQ[If[useThermal, nlTest["symTh"], nlTest["symFull"]]]];
nlIntegrand = If[useThermal, integNL12Th, integNL12];
Print["  -> integrand: ", If[useThermal, "integNL12Th (thermal-only, no zero-point noise)",
                                        "integNL12 (original form)"],
      "   v-domain: ", If[useHalf, "(0,1/2) x 2", "(0,1)"]];

(* ---------------------------------------------------------------------------- *)
(*  [G3] breakpoints + the per-(r2,w) integral                                   *)
(* ---------------------------------------------------------------------------- *)
vNodesNL[w_?NumericQ, hi_?NumericQ] := Module[{pts, vP, hw},
   pts = Flatten[Table[m*kB*tt/(hbar*om001*w), {tt, {temp1, temp2}}, {m, nlThermalBP}]];
   vP = wRes/w;  hw = bpHalfWidths*etaRel/w;
   pts = Join[pts, {vP - hw, vP, vP + hw, 1 - vP - hw, 1 - vP, 1 - vP + hw}];
   pts = Select[N[pts], 0 < # < hi &];
   Join[{0.}, DeleteDuplicates[Sort[pts], Abs[#1 - #2] < 1.*^-12 &], {N[hi]}]];

optsNL = {AccuracyGoal -> nlAcc, PrecisionGoal -> nlPrec, MaxRecursion -> nlMaxRec,
   Method -> {"GlobalAdaptive", "SingularityHandler" -> "IMT", "MaxErrorIncreases" -> nlMaxErrInc}};

(* returns {value, flag};  flag 0 = clean, 1 = NIntegrate warned, 2 = no number *)
intNL12W1v41[r2_?NumericQ, w_?NumericQ] := Module[{val, flag = 0, hi, fac, vv},
   hi  = If[useHalf, 0.5, 1.];
   fac = If[useHalf, 2., 1.];
   Quiet[Check[
      val = w*fac*NIntegrate[nlIntegrand[r2, w, vv*w, temp1, temp2],
         Evaluate[Join[{vv}, vNodesNL[w, hi]]], Evaluate[Sequence @@ optsNL]],
      flag = 1]];
   If[!NumericQ[val] || !TrueQ[Element[val, Reals]], val = Indeterminate; flag = 2];
   {val, flag}];

(* ---------------------------------------------------------------------------- *)
(*  [G6] CHECKPOINT FILES                                                        *)
(* ---------------------------------------------------------------------------- *)
nlRowDir = FileNameJoin[{outDir, "NL_rows"}];
If[!DirectoryQ[nlRowDir], Quiet[CreateDirectory[nlRowDir]]];
nlLogFile = FileNameJoin[{nlRowDir, "progress_task" <> ToString[nlTaskID] <> ".log"}];

(* any change of grid / tolerances / physics changes the signature -> stale rows  *)
(* are recomputed instead of being mixed in                                      *)
nlSig = Hash[{"ver41", r2grid, wgrid, temp1, temp2, nlPrec, nlAcc, nlMaxRec, nlMaxErrInc,
   nlThermalBP, bpHalfWidths, useThermal, useHalf, Atilde1, Atilde2, Btilde1, Btilde2,
   eta1, eta2, om001, om002, k01, k02}];

rowFile[i_Integer] := FileNameJoin[{nlRowDir, "row_" <> IntegerString[i, 10, 3] <> ".m"}];
readRow[i_Integer] := If[!FileExistsQ[rowFile[i]], $Failed,
   Quiet@Check[With[{d = Get[rowFile[i]]},
      If[ListQ[d] && Length[d] == 3 && d[[1]] === nlSig && d[[2]] === i &&
         Length[d[[3]]] == Length[wgrid], d[[3]], $Failed]], $Failed]];

logLine[s__] := Module[{str, h},
   str = DateString[] <> "  " <> StringJoin[If[StringQ[#], #, ToString[#, InputForm]] & /@ {s}];
   Print[str];
   Quiet@Check[h = OpenAppend[nlLogFile]; WriteString[h, str <> "\n"]; Close[h], Null]];

(* ---------------------------------------------------------------------------- *)
(*  COMPUTE this task's rows (round-robin over activeRows), resuming             *)
(* ---------------------------------------------------------------------------- *)
If[MemberQ[{"all", "compute"}, nlMode],
   Module[{myRows, todo, tStart, nDone = 0, estRow},
      myRows = Select[activeRows, Mod[# - 1, nlNTasks] == nlTaskID &];
      todo   = Select[myRows, readRow[#] === $Failed &];
      ensureKernels[];
      DistributeDefinitions[intNL12W1v41, nlIntegrand, integNL12Th, integNL12parts, integNL12,
         TripleKaTh, TripleKa, TripleKa0, nOccT, cothT, omOf, vNodesNL, optsNL, useHalf,
         geoData, alphaTens1, alphaTens2, r1n, r2n, EffR2, hbar, kB, om001,
         temp1, temp2, wRes, etaRel, bpHalfWidths, nlThermalBP];
      estRow = N[Length[wgrid]*150*nlTest["tEval"]/Max[1, Length[Kernels[]]]];
      logLine["task ", nlTaskID, "/", nlNTasks, ": ", Length[myRows], " rows assigned, ",
              Length[myRows] - Length[todo], " already on disk, ", Length[todo], " to do.  ",
              "rough estimate ", Round[estRow], " s per row (", Length[Kernels[]], " kernels)"];
      tStart = AbsoluteTime[];
      Do[
         Module[{res, t, tmp, nFlag},
            t = First@AbsoluteTiming[
               res = With[{r2v = r2grid[[i]]},
                  ParallelTable[intNL12W1v41[r2v, w], {w, wgrid}, Method -> "FinestGrained"]]];
            tmp = rowFile[i] <> ".tmp" <> ToString[nlTaskID];
            Put[{nlSig, i, res}, tmp];
            If[FileExistsQ[rowFile[i]], DeleteFile[rowFile[i]]];
            RenameFile[tmp, rowFile[i]];
            nDone++;
            nFlag = Count[res[[All, 2]], x_ /; x != 0];
            logLine["row ", i, "  R = ", ToString[NumberForm[r2ToAng[r2grid[[i]]], {7, 3}]], " A  done in ", Round[t], " s",
                    "   warned points: ", nFlag,
                    "   (", nDone, "/", Length[todo], ", ETA ",
                    Round[(AbsoluteTime[] - tStart)/nDone*(Length[todo] - nDone)/60.], " min)"]],
         {i, todo}];
      logLine["task ", nlTaskID, " finished its rows."]]];

(* ---------------------------------------------------------------------------- *)
(*  ASSEMBLE  (NL_MODE = all or assemble): read every row file -> raw -> CSVs    *)
(* ---------------------------------------------------------------------------- *)
If[MemberQ[{"all", "assemble"}, nlMode],
   Module[{loaded, missing, rawAll, flagged},
      loaded  = AssociationMap[readRow, activeRows];
      missing = Select[activeRows, loaded[#] === $Failed &];
      If[missing =!= {},
         Print["*** WARNING: ", Length[missing], " of ", Length[activeRows],
               " rows are missing or stale; exporting what exists.  Missing rows: ", missing];
         Print["    Resubmit the compute job(s); finished rows are kept and skipped."]];
      rawAll = Flatten[Table[
         If[loaded[i] === $Failed, {},
            MapThread[{r2grid[[i]], #1, #2[[1]], #2[[2]]} &, {wgrid, loaded[i]}]],
         {i, activeRows}], 1];
      flagged = Select[rawAll, #[[4]] != 0 &];
      raw = Select[rawAll, NumericQ[#[[3]]] && (TrueQ[keepFlagged] || #[[4]] == 0) &][[All, 1 ;; 3]];
      raw = Select[raw, Element[#[[3]], Reals] &];
      raw = N[raw];
      Print["integNL12 usable points: ", Length[raw], " / ", Length[r2grid]*Length[wgrid],
            "   (warned: ", Length[flagged], ", skipped singular rows: ", Length[singularRows]*Length[wgrid],
            ", missing rows: ", Length[missing]*Length[wgrid], ")"];
      Export[FileNameJoin[{outDir, "integNL12_flagged.csv"}],
         Prepend[{r2ToAng[#[[1]]], EofW[#[[2]]], #[[3]], #[[4]]} & /@ flagged,
                 {"R_Angstrom", "E_eV", "integNL12", "flag(1=warned,2=failed)"}], "CSV"]];

   If[Length[raw] == 0,
      Print["*** NO USABLE SAMPLES - check the self-test output, NL_rows/*.log and integNL12_flagged.csv ***"];
      Abort[]];

   (* ---------------------------------------------------------------------------- *)
   (*  SPECKLE DETECTOR  (verify the speedups did not reintroduce noise)            *)
   (* ---------------------------------------------------------------------------- *)
   flipsAlongR2[w_] := Module[{col, s},
      col = SortBy[Select[raw, #[[2]] == w &], First];
      If[Length[col] < 3, Return[0]];
      s = Select[Sign /@ col[[All, 3]], # != 0 &];
      If[Length[s] < 2, Return[0]];
      Count[Differences[s], x_ /; x != 0]];
   worstRows = TakeLargestBy[{flipsAlongR2[#], EofW[#]} & /@ wgrid, First, 5];
   Print["worst rows {sign flips along r2, hbar*w eV}: ", worstRows];
   Print["  1-2 flips = real boundary.  Many = noisy row: check integNL12_flagged.csv at that energy."];

   (* ---------------------------------------------------------------------------- *)
   (*  SIGNED-LOG floor : shared & absolute (from COLORBAR CONTROL PANEL)           *)
   (*  (auto-fit / color-normalisation / plotting removed - export-only version.)   *)
   (*  The CSV export below reads the full sampled array `raw` and this `cFloor`.   *)
   (* ---------------------------------------------------------------------------- *)
   cFloor = cFloorNL;                 (* = cFloorGlobal, identical to the integL12 map *)
   zmax   = zmaxGlobal;               (* shared color-axis limit *)
   With[{vv = raw[[All, 3]]},
     Print["integNL12 raw min/max : ", ScientificForm[MinMax[vv], 3],
           "   cFloor = ", ScientificForm[cFloor, 3],
           "   color range = [", NumberForm[-zmax, 4], ", ", NumberForm[zmax, 4], "]"]];

   (* ---- colorbar metadata : same identical scale as the integL12 map ---- *)
   Export[FileNameJoin[{outDir, "integNL12_colorbar.csv"}],
      {{"quantity", "value"}, {"cFloor", cFloor},
       {"color_lo", -zmax}, {"color_hi", zmax}}, "CSV"];


   (* ============================================================================ *)
   (*  DATA EXPORT FOR ORIGINLAB  --  integNL12 heat map                           *)
   (*  Reuses the sampled array 'raw' (no recomputation).                          *)
   (*                                                                              *)
   (*  The NL frequency grid is NON-uniform (log grid + resonance refinement), so  *)
   (*  the primary export is the long/XYZ format that OriginLab grids for you      *)
   (*  (Plot > Contour > "XYZ Contour", or make a virtual matrix).                 *)
   (*                                                                              *)
   (*    integNL12_XYZ.csv               R(Ang), E(eV), raw, signedlog             *)
   (*    integNL12_matrix_signedlog.csv  interpolated regular log-log grid         *)
   (*                                    (best-effort; use the XYZ file if this     *)
   (*                                     one is skipped)                          *)
   (* ============================================================================ *)
   Module[{slOf, xyzNL, dataRE, ipSL, xr, ye, nx = 220, ny = 220, xs, ys, mSL},
     slOf[v_] := Sign[v]*Log10[1 + Abs[v]/cFloor];   (* cFloor here = cFloorNL *)

     (* (1) LONG / XYZ  -- raw = {r2(norm), w(norm), value}; to physical units *)
     xyzNL = {r2ToAng[#[[1]]], EofW[#[[2]]], #[[3]], slOf[#[[3]]]} & /@ raw;
     xyzNL = SortBy[xyzNL, {First, #[[2]] &}];
     Export[FileNameJoin[{outDir, "integNL12_XYZ.csv"}],
        Prepend[xyzNL, {"R_Angstrom", "E_eV", "integNL12", "signedlog"}], "CSV"];

     (* (2) interpolated regular matrix in log10-log10 coords (best-effort) *)
     Quiet@Check[
       dataRE = {Log10[r2ToAng[#[[1]]]], Log10[EofW[#[[2]]]], slOf[#[[3]]]} & /@ raw;
       ipSL = Interpolation[dataRE, InterpolationOrder -> 1];
       {xr, ye} = {MinMax[dataRE[[All, 1]]], MinMax[dataRE[[All, 2]]]};
       xs = N@Subdivide[xr[[1]], xr[[2]], nx - 1];
       ys = N@Subdivide[ye[[1]], ye[[2]], ny - 1];
       mSL = Table[Quiet@Check[ipSL[x, y], ""], {y, ys}, {x, xs}];
       Export[FileNameJoin[{outDir, "integNL12_matrix_signedlog.csv"}],
          Prepend[MapThread[Prepend, {mSL, ys}], Prepend[xs, "logE_logR"]], "CSV"];
       Print["  wrote integNL12_matrix_signedlog.csv (interpolated regular grid)"],
       Print["  [note] interpolated matrix skipped (grid not a tensor product); ",
             "use integNL12_XYZ.csv with Origin's XYZ gridding."]];

     Print["OriginLab export (integNL12) -> ", outDir];
     Print["  integNL12_XYZ.csv  (primary)"];
     Print["  signed-log floor cFloor = ", ScientificForm[cFloor, 3]];
   ];
   Null
];

