݂$
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
q
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.9.12v2.9.0-18-gd8ce9f9c3018?? 
?
Adam/fm_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*'
shared_nameAdam/fm_layer/kernel/v
?
*Adam/fm_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fm_layer/kernel/v*
_output_shapes
:	?
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/fm_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*'
shared_nameAdam/fm_layer/kernel/m
?
*Adam/fm_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fm_layer/kernel/m*
_output_shapes
:	?
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:?*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:?*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:?*
dtype0
y
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_positives_1
r
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes	
:?*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name630*
value_dtype0	
m
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name594*
value_dtype0	
m
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name558*
value_dtype0	
m
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name522*
value_dtype0	
m
hash_table_4HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name486*
value_dtype0	
m
hash_table_5HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name408*
value_dtype0	
m
hash_table_6HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name372*
value_dtype0	
m
hash_table_7HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name336*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
{
fm_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
* 
shared_namefm_layer/kernel
t
#fm_layer/kernel/Read/ReadVariableOpReadVariableOpfm_layer/kernel*
_output_shapes
:	?
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_4Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_5Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_6Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_7Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Const_8Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_9Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_10Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_11Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_12Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_13Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_14Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_15Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_16Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_17Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_18Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_19Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_20Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_21Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
Const_22Const*
_output_shapes
:*
dtype0*?
value?B?B	Film-NoirBActionB	AdventureBHorrorBRomanceBWarBComedyBWesternBDocumentaryBSci-FiBDramaBThrillerBCrimeBFantasyB	AnimationBIMAXBMysteryBChildrenBMusical
?
Const_23Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                               
?
StatefulPartitionedCallStatefulPartitionedCallhash_table_7Const_8Const_9*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12225
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_6Const_10Const_11*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12233
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_5Const_12Const_13*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12241
?
StatefulPartitionedCall_3StatefulPartitionedCallhash_table_4Const_14Const_15*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12249
?
StatefulPartitionedCall_4StatefulPartitionedCallhash_table_3Const_16Const_17*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12257
?
StatefulPartitionedCall_5StatefulPartitionedCallhash_table_2Const_18Const_19*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12265
?
StatefulPartitionedCall_6StatefulPartitionedCallhash_table_1Const_20Const_21*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12273
?
StatefulPartitionedCall_7StatefulPartitionedCall
hash_tableConst_22Const_23*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_12281
?
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6^StatefulPartitionedCall_7
?G
Const_24Const"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?G B?F
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-0
layer-16
layer_with_weights-1
layer-17
layer-18
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_feature_columns
%
_resources* 
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel*
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 

,0
-1
42*

,0
-1
42*

A0
B1* 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
* 
?
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rate,m?-m?4m?,v?-v?4v?*

Userving_default* 
* 
* 
* 
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

[trace_0
\trace_1* 

]trace_0
^trace_1* 
* 
?
_movieGenre1
`movieGenre2
amovieGenre3
b
userGenre1
c
userGenre2
d
userGenre3
e
userGenre4
f
userGenre5* 

,0
-1*

,0
-1*

A0
B1* 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

ltrace_0* 

mtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

40*

40*
* 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
_Y
VARIABLE_VALUEfm_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

ztrace_0* 

{trace_0* 
* 
* 
* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*
$
?0
?1
?2
?3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?movieGenre1_lookup* 

?movieGenre2_lookup* 

?movieGenre3_lookup* 

?userGenre1_lookup* 

?userGenre2_lookup* 

?userGenre3_lookup* 

?userGenre4_lookup* 

?userGenre5_lookup* 
* 
* 
* 

A0
B1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
z
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives*
z
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives*
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
?0
?1
?2
?3*

?	variables*
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/fm_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/fm_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
serving_default_movieAvgRatingPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_movieGenre1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_movieGenre2Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_movieGenre3Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
{
 serving_default_movieRatingCountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
|
!serving_default_movieRatingStddevPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_releaseYearPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_userAvgRatingPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_userGenre1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_userGenre2Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_userGenre3Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_userGenre4Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_userGenre5Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_userRatingCountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
{
 serving_default_userRatingStddevPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_8StatefulPartitionedCallserving_default_movieAvgRatingserving_default_movieGenre1serving_default_movieGenre2serving_default_movieGenre3 serving_default_movieRatingCount!serving_default_movieRatingStddevserving_default_releaseYearserving_default_userAvgRatingserving_default_userGenre1serving_default_userGenre2serving_default_userGenre3serving_default_userGenre4serving_default_userGenre5serving_default_userRatingCount serving_default_userRatingStddevhash_table_7Consthash_table_6Const_1hash_table_5Const_2hash_table_4Const_3hash_table_3Const_4hash_table_2Const_5hash_table_1Const_6
hash_tableConst_7dense/kernel
dense/biasfm_layer/kernel*-
Tin&
$2"								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
 !*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_10467
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_9StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp#fm_layer/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp*Adam/fm_layer/kernel/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp*Adam/fm_layer/kernel/v/Read/ReadVariableOpConst_24*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_12436
?
StatefulPartitionedCall_10StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasfm_layer/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcounttrue_positives_1true_negatives_1false_positives_1false_negatives_1true_positivestrue_negativesfalse_positivesfalse_negativesAdam/dense/kernel/mAdam/dense/bias/mAdam/fm_layer/kernel/mAdam/dense/kernel/vAdam/dense/bias/vAdam/fm_layer/kernel/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_12524??
?
,
__inference__destroyer_12163
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
__inference__creator_12150
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name522*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
,
__inference__destroyer_12145
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
__inference_loss_fn_0_12062G
4dense_kernel_regularizer_abs_readvariableop_resource:	?
identity??+dense/kernel/Regularizer/Abs/ReadVariableOp?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: t
NoOpNoOp,^dense/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp
??
?
I__inference_dense_features_layer_call_and_return_conditional_losses_11670
features_movieavgrating
features_moviegenre1
features_moviegenre2
features_moviegenre3
features_movieratingcount
features_movieratingstddev
features_releaseyear
features_useravgrating
features_usergenre1
features_usergenre2
features_usergenre3
features_usergenre4
features_usergenre5
features_userratingcount
features_userratingstddevD
@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	
identity??3movieGenre1_indicator/None_Lookup/LookupTableFindV2?3movieGenre2_indicator/None_Lookup/LookupTableFindV2?3movieGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre1_indicator/None_Lookup/LookupTableFindV2?2userGenre2_indicator/None_Lookup/LookupTableFindV2?2userGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre4_indicator/None_Lookup/LookupTableFindV2?2userGenre5_indicator/None_Lookup/LookupTableFindV2h
movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieAvgRating/ExpandDims
ExpandDimsfeatures_movieavgrating&movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????f
movieAvgRating/ShapeShape"movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:l
"movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieAvgRating/strided_sliceStridedSlicemovieAvgRating/Shape:output:0+movieAvgRating/strided_slice/stack:output:0-movieAvgRating/strided_slice/stack_1:output:0-movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieAvgRating/Reshape/shapePack%movieAvgRating/strided_slice:output:0'movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieAvgRating/ReshapeReshape"movieAvgRating/ExpandDims:output:0%movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre1_indicator/ExpandDims
ExpandDimsfeatures_moviegenre1-movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre1_indicator/to_sparse_input/NotEqualNotEqual)movieGenre1_indicator/ExpandDims:output:0=movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre1_indicator/to_sparse_input/indicesWhere2movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre1_indicator/to_sparse_input/valuesGatherNd)movieGenre1_indicator/ExpandDims:output:05movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre1_indicator/to_sparse_input/dense_shapeShape)movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre1_indicator/to_sparse_input/values:output:0Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre1_indicator/SparseToDenseSparseToDense5movieGenre1_indicator/to_sparse_input/indices:index:0:movieGenre1_indicator/to_sparse_input/dense_shape:output:0<movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre1_indicator/one_hotOneHot+movieGenre1_indicator/SparseToDense:dense:0,movieGenre1_indicator/one_hot/depth:output:0,movieGenre1_indicator/one_hot/Const:output:0.movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre1_indicator/SumSum&movieGenre1_indicator/one_hot:output:04movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre1_indicator/ShapeShape"movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre1_indicator/strided_sliceStridedSlice$movieGenre1_indicator/Shape:output:02movieGenre1_indicator/strided_slice/stack:output:04movieGenre1_indicator/strided_slice/stack_1:output:04movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre1_indicator/Reshape/shapePack,movieGenre1_indicator/strided_slice:output:0.movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre1_indicator/ReshapeReshape"movieGenre1_indicator/Sum:output:0,movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre2_indicator/ExpandDims
ExpandDimsfeatures_moviegenre2-movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre2_indicator/to_sparse_input/NotEqualNotEqual)movieGenre2_indicator/ExpandDims:output:0=movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre2_indicator/to_sparse_input/indicesWhere2movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre2_indicator/to_sparse_input/valuesGatherNd)movieGenre2_indicator/ExpandDims:output:05movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre2_indicator/to_sparse_input/dense_shapeShape)movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre2_indicator/to_sparse_input/values:output:0Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre2_indicator/SparseToDenseSparseToDense5movieGenre2_indicator/to_sparse_input/indices:index:0:movieGenre2_indicator/to_sparse_input/dense_shape:output:0<movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre2_indicator/one_hotOneHot+movieGenre2_indicator/SparseToDense:dense:0,movieGenre2_indicator/one_hot/depth:output:0,movieGenre2_indicator/one_hot/Const:output:0.movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre2_indicator/SumSum&movieGenre2_indicator/one_hot:output:04movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre2_indicator/ShapeShape"movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre2_indicator/strided_sliceStridedSlice$movieGenre2_indicator/Shape:output:02movieGenre2_indicator/strided_slice/stack:output:04movieGenre2_indicator/strided_slice/stack_1:output:04movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre2_indicator/Reshape/shapePack,movieGenre2_indicator/strided_slice:output:0.movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre2_indicator/ReshapeReshape"movieGenre2_indicator/Sum:output:0,movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre3_indicator/ExpandDims
ExpandDimsfeatures_moviegenre3-movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre3_indicator/to_sparse_input/NotEqualNotEqual)movieGenre3_indicator/ExpandDims:output:0=movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre3_indicator/to_sparse_input/indicesWhere2movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre3_indicator/to_sparse_input/valuesGatherNd)movieGenre3_indicator/ExpandDims:output:05movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre3_indicator/to_sparse_input/dense_shapeShape)movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre3_indicator/to_sparse_input/values:output:0Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre3_indicator/SparseToDenseSparseToDense5movieGenre3_indicator/to_sparse_input/indices:index:0:movieGenre3_indicator/to_sparse_input/dense_shape:output:0<movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre3_indicator/one_hotOneHot+movieGenre3_indicator/SparseToDense:dense:0,movieGenre3_indicator/one_hot/depth:output:0,movieGenre3_indicator/one_hot/Const:output:0.movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre3_indicator/SumSum&movieGenre3_indicator/one_hot:output:04movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre3_indicator/ShapeShape"movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre3_indicator/strided_sliceStridedSlice$movieGenre3_indicator/Shape:output:02movieGenre3_indicator/strided_slice/stack:output:04movieGenre3_indicator/strided_slice/stack_1:output:04movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre3_indicator/Reshape/shapePack,movieGenre3_indicator/strided_slice:output:0.movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre3_indicator/ReshapeReshape"movieGenre3_indicator/Sum:output:0,movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingCount/ExpandDims
ExpandDimsfeatures_movieratingcount(movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
movieRatingCount/CastCast$movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????_
movieRatingCount/ShapeShapemovieRatingCount/Cast:y:0*
T0*
_output_shapes
:n
$movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingCount/strided_sliceStridedSlicemovieRatingCount/Shape:output:0-movieRatingCount/strided_slice/stack:output:0/movieRatingCount/strided_slice/stack_1:output:0/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingCount/Reshape/shapePack'movieRatingCount/strided_slice:output:0)movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingCount/ReshapeReshapemovieRatingCount/Cast:y:0'movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingStddev/ExpandDims
ExpandDimsfeatures_movieratingstddev)movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????l
movieRatingStddev/ShapeShape%movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:o
%movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingStddev/strided_sliceStridedSlice movieRatingStddev/Shape:output:0.movieRatingStddev/strided_slice/stack:output:00movieRatingStddev/strided_slice/stack_1:output:00movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingStddev/Reshape/shapePack(movieRatingStddev/strided_slice:output:0*movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingStddev/ReshapeReshape%movieRatingStddev/ExpandDims:output:0(movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
releaseYear/ExpandDims
ExpandDimsfeatures_releaseyear#releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????z
releaseYear/CastCastreleaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????U
releaseYear/ShapeShapereleaseYear/Cast:y:0*
T0*
_output_shapes
:i
releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
releaseYear/strided_sliceStridedSlicereleaseYear/Shape:output:0(releaseYear/strided_slice/stack:output:0*releaseYear/strided_slice/stack_1:output:0*releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
releaseYear/Reshape/shapePack"releaseYear/strided_slice:output:0$releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
releaseYear/ReshapeReshapereleaseYear/Cast:y:0"releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????g
userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userAvgRating/ExpandDims
ExpandDimsfeatures_useravgrating%userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????d
userAvgRating/ShapeShape!userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:k
!userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userAvgRating/strided_sliceStridedSliceuserAvgRating/Shape:output:0*userAvgRating/strided_slice/stack:output:0,userAvgRating/strided_slice/stack_1:output:0,userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userAvgRating/Reshape/shapePack$userAvgRating/strided_slice:output:0&userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userAvgRating/ReshapeReshape!userAvgRating/ExpandDims:output:0$userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre1_indicator/ExpandDims
ExpandDimsfeatures_usergenre1,userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre1_indicator/to_sparse_input/NotEqualNotEqual(userGenre1_indicator/ExpandDims:output:0<userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre1_indicator/to_sparse_input/indicesWhere1userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre1_indicator/to_sparse_input/valuesGatherNd(userGenre1_indicator/ExpandDims:output:04userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre1_indicator/to_sparse_input/dense_shapeShape(userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre1_indicator/to_sparse_input/values:output:0@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre1_indicator/SparseToDenseSparseToDense4userGenre1_indicator/to_sparse_input/indices:index:09userGenre1_indicator/to_sparse_input/dense_shape:output:0;userGenre1_indicator/None_Lookup/LookupTableFindV2:values:09userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre1_indicator/one_hotOneHot*userGenre1_indicator/SparseToDense:dense:0+userGenre1_indicator/one_hot/depth:output:0+userGenre1_indicator/one_hot/Const:output:0-userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre1_indicator/SumSum%userGenre1_indicator/one_hot:output:03userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre1_indicator/ShapeShape!userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre1_indicator/strided_sliceStridedSlice#userGenre1_indicator/Shape:output:01userGenre1_indicator/strided_slice/stack:output:03userGenre1_indicator/strided_slice/stack_1:output:03userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre1_indicator/Reshape/shapePack+userGenre1_indicator/strided_slice:output:0-userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre1_indicator/ReshapeReshape!userGenre1_indicator/Sum:output:0+userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre2_indicator/ExpandDims
ExpandDimsfeatures_usergenre2,userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre2_indicator/to_sparse_input/NotEqualNotEqual(userGenre2_indicator/ExpandDims:output:0<userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre2_indicator/to_sparse_input/indicesWhere1userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre2_indicator/to_sparse_input/valuesGatherNd(userGenre2_indicator/ExpandDims:output:04userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre2_indicator/to_sparse_input/dense_shapeShape(userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre2_indicator/to_sparse_input/values:output:0@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre2_indicator/SparseToDenseSparseToDense4userGenre2_indicator/to_sparse_input/indices:index:09userGenre2_indicator/to_sparse_input/dense_shape:output:0;userGenre2_indicator/None_Lookup/LookupTableFindV2:values:09userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre2_indicator/one_hotOneHot*userGenre2_indicator/SparseToDense:dense:0+userGenre2_indicator/one_hot/depth:output:0+userGenre2_indicator/one_hot/Const:output:0-userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre2_indicator/SumSum%userGenre2_indicator/one_hot:output:03userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre2_indicator/ShapeShape!userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre2_indicator/strided_sliceStridedSlice#userGenre2_indicator/Shape:output:01userGenre2_indicator/strided_slice/stack:output:03userGenre2_indicator/strided_slice/stack_1:output:03userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre2_indicator/Reshape/shapePack+userGenre2_indicator/strided_slice:output:0-userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre2_indicator/ReshapeReshape!userGenre2_indicator/Sum:output:0+userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre3_indicator/ExpandDims
ExpandDimsfeatures_usergenre3,userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre3_indicator/to_sparse_input/NotEqualNotEqual(userGenre3_indicator/ExpandDims:output:0<userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre3_indicator/to_sparse_input/indicesWhere1userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre3_indicator/to_sparse_input/valuesGatherNd(userGenre3_indicator/ExpandDims:output:04userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre3_indicator/to_sparse_input/dense_shapeShape(userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre3_indicator/to_sparse_input/values:output:0@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre3_indicator/SparseToDenseSparseToDense4userGenre3_indicator/to_sparse_input/indices:index:09userGenre3_indicator/to_sparse_input/dense_shape:output:0;userGenre3_indicator/None_Lookup/LookupTableFindV2:values:09userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre3_indicator/one_hotOneHot*userGenre3_indicator/SparseToDense:dense:0+userGenre3_indicator/one_hot/depth:output:0+userGenre3_indicator/one_hot/Const:output:0-userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre3_indicator/SumSum%userGenre3_indicator/one_hot:output:03userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre3_indicator/ShapeShape!userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre3_indicator/strided_sliceStridedSlice#userGenre3_indicator/Shape:output:01userGenre3_indicator/strided_slice/stack:output:03userGenre3_indicator/strided_slice/stack_1:output:03userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre3_indicator/Reshape/shapePack+userGenre3_indicator/strided_slice:output:0-userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre3_indicator/ReshapeReshape!userGenre3_indicator/Sum:output:0+userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre4_indicator/ExpandDims
ExpandDimsfeatures_usergenre4,userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre4_indicator/to_sparse_input/NotEqualNotEqual(userGenre4_indicator/ExpandDims:output:0<userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre4_indicator/to_sparse_input/indicesWhere1userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre4_indicator/to_sparse_input/valuesGatherNd(userGenre4_indicator/ExpandDims:output:04userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre4_indicator/to_sparse_input/dense_shapeShape(userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre4_indicator/to_sparse_input/values:output:0@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre4_indicator/SparseToDenseSparseToDense4userGenre4_indicator/to_sparse_input/indices:index:09userGenre4_indicator/to_sparse_input/dense_shape:output:0;userGenre4_indicator/None_Lookup/LookupTableFindV2:values:09userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre4_indicator/one_hotOneHot*userGenre4_indicator/SparseToDense:dense:0+userGenre4_indicator/one_hot/depth:output:0+userGenre4_indicator/one_hot/Const:output:0-userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre4_indicator/SumSum%userGenre4_indicator/one_hot:output:03userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre4_indicator/ShapeShape!userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre4_indicator/strided_sliceStridedSlice#userGenre4_indicator/Shape:output:01userGenre4_indicator/strided_slice/stack:output:03userGenre4_indicator/strided_slice/stack_1:output:03userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre4_indicator/Reshape/shapePack+userGenre4_indicator/strided_slice:output:0-userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre4_indicator/ReshapeReshape!userGenre4_indicator/Sum:output:0+userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre5_indicator/ExpandDims
ExpandDimsfeatures_usergenre5,userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre5_indicator/to_sparse_input/NotEqualNotEqual(userGenre5_indicator/ExpandDims:output:0<userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre5_indicator/to_sparse_input/indicesWhere1userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre5_indicator/to_sparse_input/valuesGatherNd(userGenre5_indicator/ExpandDims:output:04userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre5_indicator/to_sparse_input/dense_shapeShape(userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre5_indicator/to_sparse_input/values:output:0@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre5_indicator/SparseToDenseSparseToDense4userGenre5_indicator/to_sparse_input/indices:index:09userGenre5_indicator/to_sparse_input/dense_shape:output:0;userGenre5_indicator/None_Lookup/LookupTableFindV2:values:09userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre5_indicator/one_hotOneHot*userGenre5_indicator/SparseToDense:dense:0+userGenre5_indicator/one_hot/depth:output:0+userGenre5_indicator/one_hot/Const:output:0-userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre5_indicator/SumSum%userGenre5_indicator/one_hot:output:03userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre5_indicator/ShapeShape!userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre5_indicator/strided_sliceStridedSlice#userGenre5_indicator/Shape:output:01userGenre5_indicator/strided_slice/stack:output:03userGenre5_indicator/strided_slice/stack_1:output:03userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre5_indicator/Reshape/shapePack+userGenre5_indicator/strided_slice:output:0-userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre5_indicator/ReshapeReshape!userGenre5_indicator/Sum:output:0+userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingCount/ExpandDims
ExpandDimsfeatures_userratingcount'userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
userRatingCount/CastCast#userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????]
userRatingCount/ShapeShapeuserRatingCount/Cast:y:0*
T0*
_output_shapes
:m
#userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingCount/strided_sliceStridedSliceuserRatingCount/Shape:output:0,userRatingCount/strided_slice/stack:output:0.userRatingCount/strided_slice/stack_1:output:0.userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingCount/Reshape/shapePack&userRatingCount/strided_slice:output:0(userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingCount/ReshapeReshapeuserRatingCount/Cast:y:0&userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingStddev/ExpandDims
ExpandDimsfeatures_userratingstddev(userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ShapeShape$userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:n
$userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingStddev/strided_sliceStridedSliceuserRatingStddev/Shape:output:0-userRatingStddev/strided_slice/stack:output:0/userRatingStddev/strided_slice/stack_1:output:0/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingStddev/Reshape/shapePack'userRatingStddev/strided_slice:output:0)userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingStddev/ReshapeReshape$userRatingStddev/ExpandDims:output:0'userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2movieAvgRating/Reshape:output:0&movieGenre1_indicator/Reshape:output:0&movieGenre2_indicator/Reshape:output:0&movieGenre3_indicator/Reshape:output:0!movieRatingCount/Reshape:output:0"movieRatingStddev/Reshape:output:0releaseYear/Reshape:output:0userAvgRating/Reshape:output:0%userGenre1_indicator/Reshape:output:0%userGenre2_indicator/Reshape:output:0%userGenre3_indicator/Reshape:output:0%userGenre4_indicator/Reshape:output:0%userGenre5_indicator/Reshape:output:0 userRatingCount/Reshape:output:0!userRatingStddev/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp4^movieGenre1_indicator/None_Lookup/LookupTableFindV24^movieGenre2_indicator/None_Lookup/LookupTableFindV24^movieGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre1_indicator/None_Lookup/LookupTableFindV23^userGenre2_indicator/None_Lookup/LookupTableFindV23^userGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre4_indicator/None_Lookup/LookupTableFindV23^userGenre5_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : 2j
3movieGenre1_indicator/None_Lookup/LookupTableFindV23movieGenre1_indicator/None_Lookup/LookupTableFindV22j
3movieGenre2_indicator/None_Lookup/LookupTableFindV23movieGenre2_indicator/None_Lookup/LookupTableFindV22j
3movieGenre3_indicator/None_Lookup/LookupTableFindV23movieGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre1_indicator/None_Lookup/LookupTableFindV22userGenre1_indicator/None_Lookup/LookupTableFindV22h
2userGenre2_indicator/None_Lookup/LookupTableFindV22userGenre2_indicator/None_Lookup/LookupTableFindV22h
2userGenre3_indicator/None_Lookup/LookupTableFindV22userGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre4_indicator/None_Lookup/LookupTableFindV22userGenre4_indicator/None_Lookup/LookupTableFindV22h
2userGenre5_indicator/None_Lookup/LookupTableFindV22userGenre5_indicator/None_Lookup/LookupTableFindV2:\ X
#
_output_shapes
:?????????
1
_user_specified_namefeatures/movieAvgRating:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre1:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre2:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre3:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/movieRatingCount:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/movieRatingStddev:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/releaseYear:[W
#
_output_shapes
:?????????
0
_user_specified_namefeatures/userAvgRating:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre1:X	T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre2:X
T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre3:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre4:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre5:]Y
#
_output_shapes
:?????????
2
_user_specified_namefeatures/userRatingCount:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
`
D__inference_activation_layer_call_and_return_conditional_losses_9509

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_dense_features_layer_call_fn_11369
features_movieavgrating
features_moviegenre1
features_moviegenre2
features_moviegenre3
features_movieratingcount
features_movieratingstddev
features_releaseyear
features_useravgrating
features_usergenre1
features_usergenre2
features_usergenre3
features_usergenre4
features_usergenre5
features_userratingcount
features_userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieavgratingfeatures_moviegenre1features_moviegenre2features_moviegenre3features_movieratingcountfeatures_movieratingstddevfeatures_releaseyearfeatures_useravgratingfeatures_usergenre1features_usergenre2features_usergenre3features_usergenre4features_usergenre5features_userratingcountfeatures_userratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14**
Tin#
!2								*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_features_layer_call_and_return_conditional_losses_9965p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
#
_output_shapes
:?????????
1
_user_specified_namefeatures/movieAvgRating:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre1:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre2:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre3:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/movieRatingCount:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/movieRatingStddev:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/releaseYear:[W
#
_output_shapes
:?????????
0
_user_specified_namefeatures/userAvgRating:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre1:X	T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre2:X
T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre3:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre4:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre5:]Y
#
_output_shapes
:?????????
2
_user_specified_namefeatures/userRatingCount:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
:
__inference__creator_12204
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name630*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_122812
.table_init629_lookuptableimportv2_table_handle*
&table_init629_lookuptableimportv2_keys,
(table_init629_lookuptableimportv2_values	
identity??!table_init629/LookupTableImportV2?
!table_init629/LookupTableImportV2LookupTableImportV2.table_init629_lookuptableimportv2_table_handle&table_init629_lookuptableimportv2_keys(table_init629_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init629/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init629/LookupTableImportV2!table_init629/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_121762
.table_init557_lookuptableimportv2_table_handle*
&table_init557_lookuptableimportv2_keys,
(table_init557_lookuptableimportv2_values	
identity??!table_init557/LookupTableImportV2?
!table_init557/LookupTableImportV2LookupTableImportV2.table_init557_lookuptableimportv2_table_handle&table_init557_lookuptableimportv2_keys(table_init557_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init557/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init557/LookupTableImportV2!table_init557/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_122122
.table_init629_lookuptableimportv2_table_handle*
&table_init629_lookuptableimportv2_keys,
(table_init629_lookuptableimportv2_values	
identity??!table_init629/LookupTableImportV2?
!table_init629/LookupTableImportV2LookupTableImportV2.table_init629_lookuptableimportv2_table_handle&table_init629_lookuptableimportv2_keys(table_init629_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init629/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init629/LookupTableImportV2!table_init629/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
.__inference_dense_features_layer_call_fn_11318
features_movieavgrating
features_moviegenre1
features_moviegenre2
features_moviegenre3
features_movieratingcount
features_movieratingstddev
features_releaseyear
features_useravgrating
features_usergenre1
features_usergenre2
features_usergenre3
features_usergenre4
features_usergenre5
features_userratingcount
features_userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieavgratingfeatures_moviegenre1features_moviegenre2features_moviegenre3features_movieratingcountfeatures_movieratingstddevfeatures_releaseyearfeatures_useravgratingfeatures_usergenre1features_usergenre2features_usergenre3features_usergenre4features_usergenre5features_userratingcountfeatures_userratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14**
Tin#
!2								*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_features_layer_call_and_return_conditional_losses_9410p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
#
_output_shapes
:?????????
1
_user_specified_namefeatures/movieAvgRating:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre1:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre2:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre3:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/movieRatingCount:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/movieRatingStddev:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/releaseYear:[W
#
_output_shapes
:?????????
0
_user_specified_namefeatures/userAvgRating:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre1:X	T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre2:X
T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre3:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre4:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre5:]Y
#
_output_shapes
:?????????
2
_user_specified_namefeatures/userRatingCount:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_12127
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
C__inference_fm_layer_layer_call_and_return_conditional_losses_12029
x1
matmul_readvariableop_resource:	?

identity??MatMul/ReadVariableOp?Pow_2/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
PowPowMatMul:product:0Pow/y:output:0*
T0*'
_output_shapes
:?????????
L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_1PowxPow_1/y:output:0*
T0*(
_output_shapes
:??????????t
Pow_2/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
Pow_2PowPow_2/ReadVariableOp:value:0Pow_2/y:output:0*
T0*
_output_shapes
:	?
Z
MatMul_1MatMul	Pow_1:z:0	Pow_2:z:0*
T0*'
_output_shapes
:?????????
Y
subSubPow:z:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
SumSumsub:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Z
mulMulSum:output:0mul/y:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????u
NoOpNoOp^MatMul/ReadVariableOp^Pow_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2,
Pow_2/ReadVariableOpPow_2/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
$__inference_model_layer_call_fn_9565
movieavgrating
moviegenre1
moviegenre2
moviegenre3
movieratingcount
movieratingstddev
releaseyear
useravgrating

usergenre1

usergenre2

usergenre3

usergenre4

usergenre5
userratingcount
userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15:	?

unknown_16:

unknown_17:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovieavgratingmoviegenre1moviegenre2moviegenre3movieratingcountmovieratingstddevreleaseyearuseravgrating
usergenre1
usergenre2
usergenre3
usergenre4
usergenre5userratingcountuserratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*-
Tin&
$2"								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
 !*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_9524o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namemovieAvgRating:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre1:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre2:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre3:UQ
#
_output_shapes
:?????????
*
_user_specified_namemovieRatingCount:VR
#
_output_shapes
:?????????
+
_user_specified_namemovieRatingStddev:PL
#
_output_shapes
:?????????
%
_user_specified_namereleaseYear:RN
#
_output_shapes
:?????????
'
_user_specified_nameuserAvgRating:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre1:O	K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre2:O
K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre3:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre4:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre5:TP
#
_output_shapes
:?????????
)
_user_specified_nameuserRatingCount:UQ
#
_output_shapes
:?????????
*
_user_specified_nameuserRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
x
(__inference_fm_layer_layer_call_fn_12009
x
unknown:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_fm_layer_layer_call_and_return_conditional_losses_9492o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
:
__inference__creator_12078
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name336*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?8
?

__inference__traced_save_12436
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop.
*savev2_fm_layer_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_true_positives_1_read_readvariableop/
+savev2_true_negatives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop5
1savev2_adam_fm_layer_kernel_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop5
1savev2_adam_fm_layer_kernel_v_read_readvariableop
savev2_const_24

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop*savev2_fm_layer_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_true_positives_1_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop1savev2_adam_fm_layer_kernel_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop1savev2_adam_fm_layer_kernel_v_read_readvariableopsavev2_const_24"/device:CPU:0*
_output_shapes
 *)
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?::	?
: : : : : : : : : :?:?:?:?:?:?:?:?:	?::	?
:	?::	?
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?
:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?
:

_output_shapes
: 
?
?
__inference__initializer_120862
.table_init335_lookuptableimportv2_table_handle*
&table_init335_lookuptableimportv2_keys,
(table_init335_lookuptableimportv2_values	
identity??!table_init335/LookupTableImportV2?
!table_init335/LookupTableImportV2LookupTableImportV2.table_init335_lookuptableimportv2_table_handle&table_init335_lookuptableimportv2_keys(table_init335_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init335/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init335/LookupTableImportV2!table_init335/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
??
?
H__inference_dense_features_layer_call_and_return_conditional_losses_9965
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14D
@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	
identity??3movieGenre1_indicator/None_Lookup/LookupTableFindV2?3movieGenre2_indicator/None_Lookup/LookupTableFindV2?3movieGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre1_indicator/None_Lookup/LookupTableFindV2?2userGenre2_indicator/None_Lookup/LookupTableFindV2?2userGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre4_indicator/None_Lookup/LookupTableFindV2?2userGenre5_indicator/None_Lookup/LookupTableFindV2h
movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieAvgRating/ExpandDims
ExpandDimsfeatures&movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????f
movieAvgRating/ShapeShape"movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:l
"movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieAvgRating/strided_sliceStridedSlicemovieAvgRating/Shape:output:0+movieAvgRating/strided_slice/stack:output:0-movieAvgRating/strided_slice/stack_1:output:0-movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieAvgRating/Reshape/shapePack%movieAvgRating/strided_slice:output:0'movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieAvgRating/ReshapeReshape"movieAvgRating/ExpandDims:output:0%movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre1_indicator/ExpandDims
ExpandDims
features_1-movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre1_indicator/to_sparse_input/NotEqualNotEqual)movieGenre1_indicator/ExpandDims:output:0=movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre1_indicator/to_sparse_input/indicesWhere2movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre1_indicator/to_sparse_input/valuesGatherNd)movieGenre1_indicator/ExpandDims:output:05movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre1_indicator/to_sparse_input/dense_shapeShape)movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre1_indicator/to_sparse_input/values:output:0Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre1_indicator/SparseToDenseSparseToDense5movieGenre1_indicator/to_sparse_input/indices:index:0:movieGenre1_indicator/to_sparse_input/dense_shape:output:0<movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre1_indicator/one_hotOneHot+movieGenre1_indicator/SparseToDense:dense:0,movieGenre1_indicator/one_hot/depth:output:0,movieGenre1_indicator/one_hot/Const:output:0.movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre1_indicator/SumSum&movieGenre1_indicator/one_hot:output:04movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre1_indicator/ShapeShape"movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre1_indicator/strided_sliceStridedSlice$movieGenre1_indicator/Shape:output:02movieGenre1_indicator/strided_slice/stack:output:04movieGenre1_indicator/strided_slice/stack_1:output:04movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre1_indicator/Reshape/shapePack,movieGenre1_indicator/strided_slice:output:0.movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre1_indicator/ReshapeReshape"movieGenre1_indicator/Sum:output:0,movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre2_indicator/ExpandDims
ExpandDims
features_2-movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre2_indicator/to_sparse_input/NotEqualNotEqual)movieGenre2_indicator/ExpandDims:output:0=movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre2_indicator/to_sparse_input/indicesWhere2movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre2_indicator/to_sparse_input/valuesGatherNd)movieGenre2_indicator/ExpandDims:output:05movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre2_indicator/to_sparse_input/dense_shapeShape)movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre2_indicator/to_sparse_input/values:output:0Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre2_indicator/SparseToDenseSparseToDense5movieGenre2_indicator/to_sparse_input/indices:index:0:movieGenre2_indicator/to_sparse_input/dense_shape:output:0<movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre2_indicator/one_hotOneHot+movieGenre2_indicator/SparseToDense:dense:0,movieGenre2_indicator/one_hot/depth:output:0,movieGenre2_indicator/one_hot/Const:output:0.movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre2_indicator/SumSum&movieGenre2_indicator/one_hot:output:04movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre2_indicator/ShapeShape"movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre2_indicator/strided_sliceStridedSlice$movieGenre2_indicator/Shape:output:02movieGenre2_indicator/strided_slice/stack:output:04movieGenre2_indicator/strided_slice/stack_1:output:04movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre2_indicator/Reshape/shapePack,movieGenre2_indicator/strided_slice:output:0.movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre2_indicator/ReshapeReshape"movieGenre2_indicator/Sum:output:0,movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre3_indicator/ExpandDims
ExpandDims
features_3-movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre3_indicator/to_sparse_input/NotEqualNotEqual)movieGenre3_indicator/ExpandDims:output:0=movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre3_indicator/to_sparse_input/indicesWhere2movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre3_indicator/to_sparse_input/valuesGatherNd)movieGenre3_indicator/ExpandDims:output:05movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre3_indicator/to_sparse_input/dense_shapeShape)movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre3_indicator/to_sparse_input/values:output:0Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre3_indicator/SparseToDenseSparseToDense5movieGenre3_indicator/to_sparse_input/indices:index:0:movieGenre3_indicator/to_sparse_input/dense_shape:output:0<movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre3_indicator/one_hotOneHot+movieGenre3_indicator/SparseToDense:dense:0,movieGenre3_indicator/one_hot/depth:output:0,movieGenre3_indicator/one_hot/Const:output:0.movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre3_indicator/SumSum&movieGenre3_indicator/one_hot:output:04movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre3_indicator/ShapeShape"movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre3_indicator/strided_sliceStridedSlice$movieGenre3_indicator/Shape:output:02movieGenre3_indicator/strided_slice/stack:output:04movieGenre3_indicator/strided_slice/stack_1:output:04movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre3_indicator/Reshape/shapePack,movieGenre3_indicator/strided_slice:output:0.movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre3_indicator/ReshapeReshape"movieGenre3_indicator/Sum:output:0,movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingCount/ExpandDims
ExpandDims
features_4(movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
movieRatingCount/CastCast$movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????_
movieRatingCount/ShapeShapemovieRatingCount/Cast:y:0*
T0*
_output_shapes
:n
$movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingCount/strided_sliceStridedSlicemovieRatingCount/Shape:output:0-movieRatingCount/strided_slice/stack:output:0/movieRatingCount/strided_slice/stack_1:output:0/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingCount/Reshape/shapePack'movieRatingCount/strided_slice:output:0)movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingCount/ReshapeReshapemovieRatingCount/Cast:y:0'movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingStddev/ExpandDims
ExpandDims
features_5)movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????l
movieRatingStddev/ShapeShape%movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:o
%movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingStddev/strided_sliceStridedSlice movieRatingStddev/Shape:output:0.movieRatingStddev/strided_slice/stack:output:00movieRatingStddev/strided_slice/stack_1:output:00movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingStddev/Reshape/shapePack(movieRatingStddev/strided_slice:output:0*movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingStddev/ReshapeReshape%movieRatingStddev/ExpandDims:output:0(movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
releaseYear/ExpandDims
ExpandDims
features_6#releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????z
releaseYear/CastCastreleaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????U
releaseYear/ShapeShapereleaseYear/Cast:y:0*
T0*
_output_shapes
:i
releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
releaseYear/strided_sliceStridedSlicereleaseYear/Shape:output:0(releaseYear/strided_slice/stack:output:0*releaseYear/strided_slice/stack_1:output:0*releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
releaseYear/Reshape/shapePack"releaseYear/strided_slice:output:0$releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
releaseYear/ReshapeReshapereleaseYear/Cast:y:0"releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????g
userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userAvgRating/ExpandDims
ExpandDims
features_7%userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????d
userAvgRating/ShapeShape!userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:k
!userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userAvgRating/strided_sliceStridedSliceuserAvgRating/Shape:output:0*userAvgRating/strided_slice/stack:output:0,userAvgRating/strided_slice/stack_1:output:0,userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userAvgRating/Reshape/shapePack$userAvgRating/strided_slice:output:0&userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userAvgRating/ReshapeReshape!userAvgRating/ExpandDims:output:0$userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre1_indicator/ExpandDims
ExpandDims
features_8,userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre1_indicator/to_sparse_input/NotEqualNotEqual(userGenre1_indicator/ExpandDims:output:0<userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre1_indicator/to_sparse_input/indicesWhere1userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre1_indicator/to_sparse_input/valuesGatherNd(userGenre1_indicator/ExpandDims:output:04userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre1_indicator/to_sparse_input/dense_shapeShape(userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre1_indicator/to_sparse_input/values:output:0@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre1_indicator/SparseToDenseSparseToDense4userGenre1_indicator/to_sparse_input/indices:index:09userGenre1_indicator/to_sparse_input/dense_shape:output:0;userGenre1_indicator/None_Lookup/LookupTableFindV2:values:09userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre1_indicator/one_hotOneHot*userGenre1_indicator/SparseToDense:dense:0+userGenre1_indicator/one_hot/depth:output:0+userGenre1_indicator/one_hot/Const:output:0-userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre1_indicator/SumSum%userGenre1_indicator/one_hot:output:03userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre1_indicator/ShapeShape!userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre1_indicator/strided_sliceStridedSlice#userGenre1_indicator/Shape:output:01userGenre1_indicator/strided_slice/stack:output:03userGenre1_indicator/strided_slice/stack_1:output:03userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre1_indicator/Reshape/shapePack+userGenre1_indicator/strided_slice:output:0-userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre1_indicator/ReshapeReshape!userGenre1_indicator/Sum:output:0+userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre2_indicator/ExpandDims
ExpandDims
features_9,userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre2_indicator/to_sparse_input/NotEqualNotEqual(userGenre2_indicator/ExpandDims:output:0<userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre2_indicator/to_sparse_input/indicesWhere1userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre2_indicator/to_sparse_input/valuesGatherNd(userGenre2_indicator/ExpandDims:output:04userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre2_indicator/to_sparse_input/dense_shapeShape(userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre2_indicator/to_sparse_input/values:output:0@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre2_indicator/SparseToDenseSparseToDense4userGenre2_indicator/to_sparse_input/indices:index:09userGenre2_indicator/to_sparse_input/dense_shape:output:0;userGenre2_indicator/None_Lookup/LookupTableFindV2:values:09userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre2_indicator/one_hotOneHot*userGenre2_indicator/SparseToDense:dense:0+userGenre2_indicator/one_hot/depth:output:0+userGenre2_indicator/one_hot/Const:output:0-userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre2_indicator/SumSum%userGenre2_indicator/one_hot:output:03userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre2_indicator/ShapeShape!userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre2_indicator/strided_sliceStridedSlice#userGenre2_indicator/Shape:output:01userGenre2_indicator/strided_slice/stack:output:03userGenre2_indicator/strided_slice/stack_1:output:03userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre2_indicator/Reshape/shapePack+userGenre2_indicator/strided_slice:output:0-userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre2_indicator/ReshapeReshape!userGenre2_indicator/Sum:output:0+userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre3_indicator/ExpandDims
ExpandDimsfeatures_10,userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre3_indicator/to_sparse_input/NotEqualNotEqual(userGenre3_indicator/ExpandDims:output:0<userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre3_indicator/to_sparse_input/indicesWhere1userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre3_indicator/to_sparse_input/valuesGatherNd(userGenre3_indicator/ExpandDims:output:04userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre3_indicator/to_sparse_input/dense_shapeShape(userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre3_indicator/to_sparse_input/values:output:0@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre3_indicator/SparseToDenseSparseToDense4userGenre3_indicator/to_sparse_input/indices:index:09userGenre3_indicator/to_sparse_input/dense_shape:output:0;userGenre3_indicator/None_Lookup/LookupTableFindV2:values:09userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre3_indicator/one_hotOneHot*userGenre3_indicator/SparseToDense:dense:0+userGenre3_indicator/one_hot/depth:output:0+userGenre3_indicator/one_hot/Const:output:0-userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre3_indicator/SumSum%userGenre3_indicator/one_hot:output:03userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre3_indicator/ShapeShape!userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre3_indicator/strided_sliceStridedSlice#userGenre3_indicator/Shape:output:01userGenre3_indicator/strided_slice/stack:output:03userGenre3_indicator/strided_slice/stack_1:output:03userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre3_indicator/Reshape/shapePack+userGenre3_indicator/strided_slice:output:0-userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre3_indicator/ReshapeReshape!userGenre3_indicator/Sum:output:0+userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre4_indicator/ExpandDims
ExpandDimsfeatures_11,userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre4_indicator/to_sparse_input/NotEqualNotEqual(userGenre4_indicator/ExpandDims:output:0<userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre4_indicator/to_sparse_input/indicesWhere1userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre4_indicator/to_sparse_input/valuesGatherNd(userGenre4_indicator/ExpandDims:output:04userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre4_indicator/to_sparse_input/dense_shapeShape(userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre4_indicator/to_sparse_input/values:output:0@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre4_indicator/SparseToDenseSparseToDense4userGenre4_indicator/to_sparse_input/indices:index:09userGenre4_indicator/to_sparse_input/dense_shape:output:0;userGenre4_indicator/None_Lookup/LookupTableFindV2:values:09userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre4_indicator/one_hotOneHot*userGenre4_indicator/SparseToDense:dense:0+userGenre4_indicator/one_hot/depth:output:0+userGenre4_indicator/one_hot/Const:output:0-userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre4_indicator/SumSum%userGenre4_indicator/one_hot:output:03userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre4_indicator/ShapeShape!userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre4_indicator/strided_sliceStridedSlice#userGenre4_indicator/Shape:output:01userGenre4_indicator/strided_slice/stack:output:03userGenre4_indicator/strided_slice/stack_1:output:03userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre4_indicator/Reshape/shapePack+userGenre4_indicator/strided_slice:output:0-userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre4_indicator/ReshapeReshape!userGenre4_indicator/Sum:output:0+userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre5_indicator/ExpandDims
ExpandDimsfeatures_12,userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre5_indicator/to_sparse_input/NotEqualNotEqual(userGenre5_indicator/ExpandDims:output:0<userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre5_indicator/to_sparse_input/indicesWhere1userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre5_indicator/to_sparse_input/valuesGatherNd(userGenre5_indicator/ExpandDims:output:04userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre5_indicator/to_sparse_input/dense_shapeShape(userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre5_indicator/to_sparse_input/values:output:0@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre5_indicator/SparseToDenseSparseToDense4userGenre5_indicator/to_sparse_input/indices:index:09userGenre5_indicator/to_sparse_input/dense_shape:output:0;userGenre5_indicator/None_Lookup/LookupTableFindV2:values:09userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre5_indicator/one_hotOneHot*userGenre5_indicator/SparseToDense:dense:0+userGenre5_indicator/one_hot/depth:output:0+userGenre5_indicator/one_hot/Const:output:0-userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre5_indicator/SumSum%userGenre5_indicator/one_hot:output:03userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre5_indicator/ShapeShape!userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre5_indicator/strided_sliceStridedSlice#userGenre5_indicator/Shape:output:01userGenre5_indicator/strided_slice/stack:output:03userGenre5_indicator/strided_slice/stack_1:output:03userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre5_indicator/Reshape/shapePack+userGenre5_indicator/strided_slice:output:0-userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre5_indicator/ReshapeReshape!userGenre5_indicator/Sum:output:0+userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingCount/ExpandDims
ExpandDimsfeatures_13'userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
userRatingCount/CastCast#userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????]
userRatingCount/ShapeShapeuserRatingCount/Cast:y:0*
T0*
_output_shapes
:m
#userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingCount/strided_sliceStridedSliceuserRatingCount/Shape:output:0,userRatingCount/strided_slice/stack:output:0.userRatingCount/strided_slice/stack_1:output:0.userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingCount/Reshape/shapePack&userRatingCount/strided_slice:output:0(userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingCount/ReshapeReshapeuserRatingCount/Cast:y:0&userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingStddev/ExpandDims
ExpandDimsfeatures_14(userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ShapeShape$userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:n
$userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingStddev/strided_sliceStridedSliceuserRatingStddev/Shape:output:0-userRatingStddev/strided_slice/stack:output:0/userRatingStddev/strided_slice/stack_1:output:0/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingStddev/Reshape/shapePack'userRatingStddev/strided_slice:output:0)userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingStddev/ReshapeReshape$userRatingStddev/ExpandDims:output:0'userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2movieAvgRating/Reshape:output:0&movieGenre1_indicator/Reshape:output:0&movieGenre2_indicator/Reshape:output:0&movieGenre3_indicator/Reshape:output:0!movieRatingCount/Reshape:output:0"movieRatingStddev/Reshape:output:0releaseYear/Reshape:output:0userAvgRating/Reshape:output:0%userGenre1_indicator/Reshape:output:0%userGenre2_indicator/Reshape:output:0%userGenre3_indicator/Reshape:output:0%userGenre4_indicator/Reshape:output:0%userGenre5_indicator/Reshape:output:0 userRatingCount/Reshape:output:0!userRatingStddev/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp4^movieGenre1_indicator/None_Lookup/LookupTableFindV24^movieGenre2_indicator/None_Lookup/LookupTableFindV24^movieGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre1_indicator/None_Lookup/LookupTableFindV23^userGenre2_indicator/None_Lookup/LookupTableFindV23^userGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre4_indicator/None_Lookup/LookupTableFindV23^userGenre5_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : 2j
3movieGenre1_indicator/None_Lookup/LookupTableFindV23movieGenre1_indicator/None_Lookup/LookupTableFindV22j
3movieGenre2_indicator/None_Lookup/LookupTableFindV23movieGenre2_indicator/None_Lookup/LookupTableFindV22j
3movieGenre3_indicator/None_Lookup/LookupTableFindV23movieGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre1_indicator/None_Lookup/LookupTableFindV22userGenre1_indicator/None_Lookup/LookupTableFindV22h
2userGenre2_indicator/None_Lookup/LookupTableFindV22userGenre2_indicator/None_Lookup/LookupTableFindV22h
2userGenre3_indicator/None_Lookup/LookupTableFindV22userGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre4_indicator/None_Lookup/LookupTableFindV22userGenre4_indicator/None_Lookup/LookupTableFindV22h
2userGenre5_indicator/None_Lookup/LookupTableFindV22userGenre5_indicator/None_Lookup/LookupTableFindV2:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:M	I
#
_output_shapes
:?????????
"
_user_specified_name
features:M
I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_dense_layer_call_and_return_conditional_losses_9466

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_122492
.table_init485_lookuptableimportv2_table_handle*
&table_init485_lookuptableimportv2_keys,
(table_init485_lookuptableimportv2_values	
identity??!table_init485/LookupTableImportV2?
!table_init485/LookupTableImportV2LookupTableImportV2.table_init485_lookuptableimportv2_table_handle&table_init485_lookuptableimportv2_keys(table_init485_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init485/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init485/LookupTableImportV2!table_init485/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?;
?
@__inference_model_layer_call_and_return_conditional_losses_10390
movieavgrating
moviegenre1
moviegenre2
moviegenre3
movieratingcount
movieratingstddev
releaseyear
useravgrating

usergenre1

usergenre2

usergenre3

usergenre4

usergenre5
userratingcount
userratingstddev
dense_features_10334
dense_features_10336	
dense_features_10338
dense_features_10340	
dense_features_10342
dense_features_10344	
dense_features_10346
dense_features_10348	
dense_features_10350
dense_features_10352	
dense_features_10354
dense_features_10356	
dense_features_10358
dense_features_10360	
dense_features_10362
dense_features_10364	
dense_10367:	?
dense_10369:!
fm_layer_10372:	?

identity??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?&dense_features/StatefulPartitionedCall? fm_layer/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallmovieavgratingmoviegenre1moviegenre2moviegenre3movieratingcountmovieratingstddevreleaseyearuseravgrating
usergenre1
usergenre2
usergenre3
usergenre4
usergenre5userratingcountuserratingstddevdense_features_10334dense_features_10336dense_features_10338dense_features_10340dense_features_10342dense_features_10344dense_features_10346dense_features_10348dense_features_10350dense_features_10352dense_features_10354dense_features_10356dense_features_10358dense_features_10360dense_features_10362dense_features_10364**
Tin#
!2								*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_features_layer_call_and_return_conditional_losses_9965?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_10367dense_10369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_9466?
 fm_layer/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0fm_layer_10372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_fm_layer_layer_call_and_return_conditional_losses_9492?
add/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0)fm_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_9502?
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_9509x
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_10367*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10369*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp'^dense_features/StatefulPartitionedCall!^fm_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2D
 fm_layer/StatefulPartitionedCall fm_layer/StatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namemovieAvgRating:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre1:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre2:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre3:UQ
#
_output_shapes
:?????????
*
_user_specified_namemovieRatingCount:VR
#
_output_shapes
:?????????
+
_user_specified_namemovieRatingStddev:PL
#
_output_shapes
:?????????
%
_user_specified_namereleaseYear:RN
#
_output_shapes
:?????????
'
_user_specified_nameuserAvgRating:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre1:O	K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre2:O
K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre3:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre4:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre5:TP
#
_output_shapes
:?????????
)
_user_specified_nameuserRatingCount:UQ
#
_output_shapes
:?????????
*
_user_specified_nameuserRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?g
?
!__inference__traced_restore_12524
file_prefix0
assignvariableop_dense_kernel:	?+
assignvariableop_1_dense_bias:5
"assignvariableop_2_fm_layer_kernel:	?
&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: $
assignvariableop_8_total_1: $
assignvariableop_9_count_1: #
assignvariableop_10_total: #
assignvariableop_11_count: 3
$assignvariableop_12_true_positives_1:	?3
$assignvariableop_13_true_negatives_1:	?4
%assignvariableop_14_false_positives_1:	?4
%assignvariableop_15_false_negatives_1:	?1
"assignvariableop_16_true_positives:	?1
"assignvariableop_17_true_negatives:	?2
#assignvariableop_18_false_positives:	?2
#assignvariableop_19_false_negatives:	?:
'assignvariableop_20_adam_dense_kernel_m:	?3
%assignvariableop_21_adam_dense_bias_m:=
*assignvariableop_22_adam_fm_layer_kernel_m:	?
:
'assignvariableop_23_adam_dense_kernel_v:	?3
%assignvariableop_24_adam_dense_bias_v:=
*assignvariableop_25_adam_fm_layer_kernel_v:	?

identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_fm_layer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_true_positives_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_true_negatives_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_false_positives_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_false_negatives_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_true_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_true_negativesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_positivesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_false_negativesIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_fm_layer_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_fm_layer_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference__initializer_121582
.table_init521_lookuptableimportv2_table_handle*
&table_init521_lookuptableimportv2_keys,
(table_init521_lookuptableimportv2_values	
identity??!table_init521/LookupTableImportV2?
!table_init521/LookupTableImportV2LookupTableImportV2.table_init521_lookuptableimportv2_table_handle&table_init521_lookuptableimportv2_keys(table_init521_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init521/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init521/LookupTableImportV2!table_init521/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
@__inference_dense_layer_call_and_return_conditional_losses_12002

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_11980

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_9466o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_10536
inputs_movieavgrating
inputs_moviegenre1
inputs_moviegenre2
inputs_moviegenre3
inputs_movieratingcount
inputs_movieratingstddev
inputs_releaseyear
inputs_useravgrating
inputs_usergenre1
inputs_usergenre2
inputs_usergenre3
inputs_usergenre4
inputs_usergenre5
inputs_userratingcount
inputs_userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15:	?

unknown_16:

unknown_17:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_movieavgratinginputs_moviegenre1inputs_moviegenre2inputs_moviegenre3inputs_movieratingcountinputs_movieratingstddevinputs_releaseyearinputs_useravgratinginputs_usergenre1inputs_usergenre2inputs_usergenre3inputs_usergenre4inputs_usergenre5inputs_userratingcountinputs_userratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*-
Tin&
$2"								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
 !*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_9524o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_nameinputs/movieAvgRating:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre1:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre2:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre3:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/movieRatingCount:]Y
#
_output_shapes
:?????????
2
_user_specified_nameinputs/movieRatingStddev:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/releaseYear:YU
#
_output_shapes
:?????????
.
_user_specified_nameinputs/userAvgRating:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre1:V	R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre2:V
R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre3:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre4:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre5:[W
#
_output_shapes
:?????????
0
_user_specified_nameinputs/userRatingCount:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?;
?
@__inference_model_layer_call_and_return_conditional_losses_10317
movieavgrating
moviegenre1
moviegenre2
moviegenre3
movieratingcount
movieratingstddev
releaseyear
useravgrating

usergenre1

usergenre2

usergenre3

usergenre4

usergenre5
userratingcount
userratingstddev
dense_features_10261
dense_features_10263	
dense_features_10265
dense_features_10267	
dense_features_10269
dense_features_10271	
dense_features_10273
dense_features_10275	
dense_features_10277
dense_features_10279	
dense_features_10281
dense_features_10283	
dense_features_10285
dense_features_10287	
dense_features_10289
dense_features_10291	
dense_10294:	?
dense_10296:!
fm_layer_10299:	?

identity??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?&dense_features/StatefulPartitionedCall? fm_layer/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallmovieavgratingmoviegenre1moviegenre2moviegenre3movieratingcountmovieratingstddevreleaseyearuseravgrating
usergenre1
usergenre2
usergenre3
usergenre4
usergenre5userratingcountuserratingstddevdense_features_10261dense_features_10263dense_features_10265dense_features_10267dense_features_10269dense_features_10271dense_features_10273dense_features_10275dense_features_10277dense_features_10279dense_features_10281dense_features_10283dense_features_10285dense_features_10287dense_features_10289dense_features_10291**
Tin#
!2								*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_features_layer_call_and_return_conditional_losses_9410?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_10294dense_10296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_9466?
 fm_layer/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0fm_layer_10299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_fm_layer_layer_call_and_return_conditional_losses_9492?
add/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0)fm_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_9502?
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_9509x
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_10294*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10296*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp'^dense_features/StatefulPartitionedCall!^fm_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2D
 fm_layer/StatefulPartitionedCall fm_layer/StatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namemovieAvgRating:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre1:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre2:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre3:UQ
#
_output_shapes
:?????????
*
_user_specified_namemovieRatingCount:VR
#
_output_shapes
:?????????
+
_user_specified_namemovieRatingStddev:PL
#
_output_shapes
:?????????
%
_user_specified_namereleaseYear:RN
#
_output_shapes
:?????????
'
_user_specified_nameuserAvgRating:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre1:O	K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre2:O
K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre3:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre4:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre5:TP
#
_output_shapes
:?????????
)
_user_specified_nameuserRatingCount:UQ
#
_output_shapes
:?????????
*
_user_specified_nameuserRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
@__inference_model_layer_call_and_return_conditional_losses_10930
inputs_movieavgrating
inputs_moviegenre1
inputs_moviegenre2
inputs_moviegenre3
inputs_movieratingcount
inputs_movieratingstddev
inputs_releaseyear
inputs_useravgrating
inputs_usergenre1
inputs_usergenre2
inputs_usergenre3
inputs_usergenre4
inputs_usergenre5
inputs_userratingcount
inputs_userratingstddevS
Odense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource::
'fm_layer_matmul_readvariableop_resource:	?

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2?Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2?Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2?fm_layer/MatMul/ReadVariableOp?fm_layer/Pow_2/ReadVariableOpw
,dense_features/movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(dense_features/movieAvgRating/ExpandDims
ExpandDimsinputs_movieavgrating5dense_features/movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
#dense_features/movieAvgRating/ShapeShape1dense_features/movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:{
1dense_features/movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_features/movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_features/movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_features/movieAvgRating/strided_sliceStridedSlice,dense_features/movieAvgRating/Shape:output:0:dense_features/movieAvgRating/strided_slice/stack:output:0<dense_features/movieAvgRating/strided_slice/stack_1:output:0<dense_features/movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-dense_features/movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/movieAvgRating/Reshape/shapePack4dense_features/movieAvgRating/strided_slice:output:06dense_features/movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%dense_features/movieAvgRating/ReshapeReshape1dense_features/movieAvgRating/ExpandDims:output:04dense_features/movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/movieGenre1_indicator/ExpandDims
ExpandDimsinputs_moviegenre1<dense_features/movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/movieGenre1_indicator/to_sparse_input/NotEqualNotEqual8dense_features/movieGenre1_indicator/ExpandDims:output:0Ldense_features/movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/movieGenre1_indicator/to_sparse_input/indicesWhereAdense_features/movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/movieGenre1_indicator/to_sparse_input/valuesGatherNd8dense_features/movieGenre1_indicator/ExpandDims:output:0Ddense_features/movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/movieGenre1_indicator/to_sparse_input/dense_shapeShape8dense_features/movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/movieGenre1_indicator/to_sparse_input/values:output:0Pdense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/movieGenre1_indicator/SparseToDenseSparseToDenseDdense_features/movieGenre1_indicator/to_sparse_input/indices:index:0Idense_features/movieGenre1_indicator/to_sparse_input/dense_shape:output:0Kdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/movieGenre1_indicator/one_hotOneHot:dense_features/movieGenre1_indicator/SparseToDense:dense:0;dense_features/movieGenre1_indicator/one_hot/depth:output:0;dense_features/movieGenre1_indicator/one_hot/Const:output:0=dense_features/movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/movieGenre1_indicator/SumSum5dense_features/movieGenre1_indicator/one_hot:output:0Cdense_features/movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/movieGenre1_indicator/ShapeShape1dense_features/movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/movieGenre1_indicator/strided_sliceStridedSlice3dense_features/movieGenre1_indicator/Shape:output:0Adense_features/movieGenre1_indicator/strided_slice/stack:output:0Cdense_features/movieGenre1_indicator/strided_slice/stack_1:output:0Cdense_features/movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/movieGenre1_indicator/Reshape/shapePack;dense_features/movieGenre1_indicator/strided_slice:output:0=dense_features/movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/movieGenre1_indicator/ReshapeReshape1dense_features/movieGenre1_indicator/Sum:output:0;dense_features/movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/movieGenre2_indicator/ExpandDims
ExpandDimsinputs_moviegenre2<dense_features/movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/movieGenre2_indicator/to_sparse_input/NotEqualNotEqual8dense_features/movieGenre2_indicator/ExpandDims:output:0Ldense_features/movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/movieGenre2_indicator/to_sparse_input/indicesWhereAdense_features/movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/movieGenre2_indicator/to_sparse_input/valuesGatherNd8dense_features/movieGenre2_indicator/ExpandDims:output:0Ddense_features/movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/movieGenre2_indicator/to_sparse_input/dense_shapeShape8dense_features/movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/movieGenre2_indicator/to_sparse_input/values:output:0Pdense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/movieGenre2_indicator/SparseToDenseSparseToDenseDdense_features/movieGenre2_indicator/to_sparse_input/indices:index:0Idense_features/movieGenre2_indicator/to_sparse_input/dense_shape:output:0Kdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/movieGenre2_indicator/one_hotOneHot:dense_features/movieGenre2_indicator/SparseToDense:dense:0;dense_features/movieGenre2_indicator/one_hot/depth:output:0;dense_features/movieGenre2_indicator/one_hot/Const:output:0=dense_features/movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/movieGenre2_indicator/SumSum5dense_features/movieGenre2_indicator/one_hot:output:0Cdense_features/movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/movieGenre2_indicator/ShapeShape1dense_features/movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/movieGenre2_indicator/strided_sliceStridedSlice3dense_features/movieGenre2_indicator/Shape:output:0Adense_features/movieGenre2_indicator/strided_slice/stack:output:0Cdense_features/movieGenre2_indicator/strided_slice/stack_1:output:0Cdense_features/movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/movieGenre2_indicator/Reshape/shapePack;dense_features/movieGenre2_indicator/strided_slice:output:0=dense_features/movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/movieGenre2_indicator/ReshapeReshape1dense_features/movieGenre2_indicator/Sum:output:0;dense_features/movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/movieGenre3_indicator/ExpandDims
ExpandDimsinputs_moviegenre3<dense_features/movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/movieGenre3_indicator/to_sparse_input/NotEqualNotEqual8dense_features/movieGenre3_indicator/ExpandDims:output:0Ldense_features/movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/movieGenre3_indicator/to_sparse_input/indicesWhereAdense_features/movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/movieGenre3_indicator/to_sparse_input/valuesGatherNd8dense_features/movieGenre3_indicator/ExpandDims:output:0Ddense_features/movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/movieGenre3_indicator/to_sparse_input/dense_shapeShape8dense_features/movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/movieGenre3_indicator/to_sparse_input/values:output:0Pdense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/movieGenre3_indicator/SparseToDenseSparseToDenseDdense_features/movieGenre3_indicator/to_sparse_input/indices:index:0Idense_features/movieGenre3_indicator/to_sparse_input/dense_shape:output:0Kdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/movieGenre3_indicator/one_hotOneHot:dense_features/movieGenre3_indicator/SparseToDense:dense:0;dense_features/movieGenre3_indicator/one_hot/depth:output:0;dense_features/movieGenre3_indicator/one_hot/Const:output:0=dense_features/movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/movieGenre3_indicator/SumSum5dense_features/movieGenre3_indicator/one_hot:output:0Cdense_features/movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/movieGenre3_indicator/ShapeShape1dense_features/movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/movieGenre3_indicator/strided_sliceStridedSlice3dense_features/movieGenre3_indicator/Shape:output:0Adense_features/movieGenre3_indicator/strided_slice/stack:output:0Cdense_features/movieGenre3_indicator/strided_slice/stack_1:output:0Cdense_features/movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/movieGenre3_indicator/Reshape/shapePack;dense_features/movieGenre3_indicator/strided_slice:output:0=dense_features/movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/movieGenre3_indicator/ReshapeReshape1dense_features/movieGenre3_indicator/Sum:output:0;dense_features/movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
.dense_features/movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*dense_features/movieRatingCount/ExpandDims
ExpandDimsinputs_movieratingcount7dense_features/movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
$dense_features/movieRatingCount/CastCast3dense_features/movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????}
%dense_features/movieRatingCount/ShapeShape(dense_features/movieRatingCount/Cast:y:0*
T0*
_output_shapes
:}
3dense_features/movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/movieRatingCount/strided_sliceStridedSlice.dense_features/movieRatingCount/Shape:output:0<dense_features/movieRatingCount/strided_slice/stack:output:0>dense_features/movieRatingCount/strided_slice/stack_1:output:0>dense_features/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/movieRatingCount/Reshape/shapePack6dense_features/movieRatingCount/strided_slice:output:08dense_features/movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/movieRatingCount/ReshapeReshape(dense_features/movieRatingCount/Cast:y:06dense_features/movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/dense_features/movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+dense_features/movieRatingStddev/ExpandDims
ExpandDimsinputs_movieratingstddev8dense_features/movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
&dense_features/movieRatingStddev/ShapeShape4dense_features/movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:~
4dense_features/movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/movieRatingStddev/strided_sliceStridedSlice/dense_features/movieRatingStddev/Shape:output:0=dense_features/movieRatingStddev/strided_slice/stack:output:0?dense_features/movieRatingStddev/strided_slice/stack_1:output:0?dense_features/movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/movieRatingStddev/Reshape/shapePack7dense_features/movieRatingStddev/strided_slice:output:09dense_features/movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/movieRatingStddev/ReshapeReshape4dense_features/movieRatingStddev/ExpandDims:output:07dense_features/movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
)dense_features/releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%dense_features/releaseYear/ExpandDims
ExpandDimsinputs_releaseyear2dense_features/releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
dense_features/releaseYear/CastCast.dense_features/releaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????s
 dense_features/releaseYear/ShapeShape#dense_features/releaseYear/Cast:y:0*
T0*
_output_shapes
:x
.dense_features/releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(dense_features/releaseYear/strided_sliceStridedSlice)dense_features/releaseYear/Shape:output:07dense_features/releaseYear/strided_slice/stack:output:09dense_features/releaseYear/strided_slice/stack_1:output:09dense_features/releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
(dense_features/releaseYear/Reshape/shapePack1dense_features/releaseYear/strided_slice:output:03dense_features/releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
"dense_features/releaseYear/ReshapeReshape#dense_features/releaseYear/Cast:y:01dense_features/releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????v
+dense_features/userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'dense_features/userAvgRating/ExpandDims
ExpandDimsinputs_useravgrating4dense_features/userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
"dense_features/userAvgRating/ShapeShape0dense_features/userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:z
0dense_features/userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_features/userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_features/userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*dense_features/userAvgRating/strided_sliceStridedSlice+dense_features/userAvgRating/Shape:output:09dense_features/userAvgRating/strided_slice/stack:output:0;dense_features/userAvgRating/strided_slice/stack_1:output:0;dense_features/userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dense_features/userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
*dense_features/userAvgRating/Reshape/shapePack3dense_features/userAvgRating/strided_slice:output:05dense_features/userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$dense_features/userAvgRating/ReshapeReshape0dense_features/userAvgRating/ExpandDims:output:03dense_features/userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre1_indicator/ExpandDims
ExpandDimsinputs_usergenre1;dense_features/userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre1_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre1_indicator/ExpandDims:output:0Kdense_features/userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre1_indicator/to_sparse_input/indicesWhere@dense_features/userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre1_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre1_indicator/ExpandDims:output:0Cdense_features/userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre1_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre1_indicator/to_sparse_input/values:output:0Odense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre1_indicator/SparseToDenseSparseToDenseCdense_features/userGenre1_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre1_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre1_indicator/one_hotOneHot9dense_features/userGenre1_indicator/SparseToDense:dense:0:dense_features/userGenre1_indicator/one_hot/depth:output:0:dense_features/userGenre1_indicator/one_hot/Const:output:0<dense_features/userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre1_indicator/SumSum4dense_features/userGenre1_indicator/one_hot:output:0Bdense_features/userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre1_indicator/ShapeShape0dense_features/userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre1_indicator/strided_sliceStridedSlice2dense_features/userGenre1_indicator/Shape:output:0@dense_features/userGenre1_indicator/strided_slice/stack:output:0Bdense_features/userGenre1_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre1_indicator/Reshape/shapePack:dense_features/userGenre1_indicator/strided_slice:output:0<dense_features/userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre1_indicator/ReshapeReshape0dense_features/userGenre1_indicator/Sum:output:0:dense_features/userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre2_indicator/ExpandDims
ExpandDimsinputs_usergenre2;dense_features/userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre2_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre2_indicator/ExpandDims:output:0Kdense_features/userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre2_indicator/to_sparse_input/indicesWhere@dense_features/userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre2_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre2_indicator/ExpandDims:output:0Cdense_features/userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre2_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre2_indicator/to_sparse_input/values:output:0Odense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre2_indicator/SparseToDenseSparseToDenseCdense_features/userGenre2_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre2_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre2_indicator/one_hotOneHot9dense_features/userGenre2_indicator/SparseToDense:dense:0:dense_features/userGenre2_indicator/one_hot/depth:output:0:dense_features/userGenre2_indicator/one_hot/Const:output:0<dense_features/userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre2_indicator/SumSum4dense_features/userGenre2_indicator/one_hot:output:0Bdense_features/userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre2_indicator/ShapeShape0dense_features/userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre2_indicator/strided_sliceStridedSlice2dense_features/userGenre2_indicator/Shape:output:0@dense_features/userGenre2_indicator/strided_slice/stack:output:0Bdense_features/userGenre2_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre2_indicator/Reshape/shapePack:dense_features/userGenre2_indicator/strided_slice:output:0<dense_features/userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre2_indicator/ReshapeReshape0dense_features/userGenre2_indicator/Sum:output:0:dense_features/userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre3_indicator/ExpandDims
ExpandDimsinputs_usergenre3;dense_features/userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre3_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre3_indicator/ExpandDims:output:0Kdense_features/userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre3_indicator/to_sparse_input/indicesWhere@dense_features/userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre3_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre3_indicator/ExpandDims:output:0Cdense_features/userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre3_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre3_indicator/to_sparse_input/values:output:0Odense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre3_indicator/SparseToDenseSparseToDenseCdense_features/userGenre3_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre3_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre3_indicator/one_hotOneHot9dense_features/userGenre3_indicator/SparseToDense:dense:0:dense_features/userGenre3_indicator/one_hot/depth:output:0:dense_features/userGenre3_indicator/one_hot/Const:output:0<dense_features/userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre3_indicator/SumSum4dense_features/userGenre3_indicator/one_hot:output:0Bdense_features/userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre3_indicator/ShapeShape0dense_features/userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre3_indicator/strided_sliceStridedSlice2dense_features/userGenre3_indicator/Shape:output:0@dense_features/userGenre3_indicator/strided_slice/stack:output:0Bdense_features/userGenre3_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre3_indicator/Reshape/shapePack:dense_features/userGenre3_indicator/strided_slice:output:0<dense_features/userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre3_indicator/ReshapeReshape0dense_features/userGenre3_indicator/Sum:output:0:dense_features/userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre4_indicator/ExpandDims
ExpandDimsinputs_usergenre4;dense_features/userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre4_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre4_indicator/ExpandDims:output:0Kdense_features/userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre4_indicator/to_sparse_input/indicesWhere@dense_features/userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre4_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre4_indicator/ExpandDims:output:0Cdense_features/userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre4_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre4_indicator/to_sparse_input/values:output:0Odense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre4_indicator/SparseToDenseSparseToDenseCdense_features/userGenre4_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre4_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre4_indicator/one_hotOneHot9dense_features/userGenre4_indicator/SparseToDense:dense:0:dense_features/userGenre4_indicator/one_hot/depth:output:0:dense_features/userGenre4_indicator/one_hot/Const:output:0<dense_features/userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre4_indicator/SumSum4dense_features/userGenre4_indicator/one_hot:output:0Bdense_features/userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre4_indicator/ShapeShape0dense_features/userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre4_indicator/strided_sliceStridedSlice2dense_features/userGenre4_indicator/Shape:output:0@dense_features/userGenre4_indicator/strided_slice/stack:output:0Bdense_features/userGenre4_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre4_indicator/Reshape/shapePack:dense_features/userGenre4_indicator/strided_slice:output:0<dense_features/userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre4_indicator/ReshapeReshape0dense_features/userGenre4_indicator/Sum:output:0:dense_features/userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre5_indicator/ExpandDims
ExpandDimsinputs_usergenre5;dense_features/userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre5_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre5_indicator/ExpandDims:output:0Kdense_features/userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre5_indicator/to_sparse_input/indicesWhere@dense_features/userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre5_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre5_indicator/ExpandDims:output:0Cdense_features/userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre5_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre5_indicator/to_sparse_input/values:output:0Odense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre5_indicator/SparseToDenseSparseToDenseCdense_features/userGenre5_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre5_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre5_indicator/one_hotOneHot9dense_features/userGenre5_indicator/SparseToDense:dense:0:dense_features/userGenre5_indicator/one_hot/depth:output:0:dense_features/userGenre5_indicator/one_hot/Const:output:0<dense_features/userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre5_indicator/SumSum4dense_features/userGenre5_indicator/one_hot:output:0Bdense_features/userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre5_indicator/ShapeShape0dense_features/userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre5_indicator/strided_sliceStridedSlice2dense_features/userGenre5_indicator/Shape:output:0@dense_features/userGenre5_indicator/strided_slice/stack:output:0Bdense_features/userGenre5_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre5_indicator/Reshape/shapePack:dense_features/userGenre5_indicator/strided_slice:output:0<dense_features/userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre5_indicator/ReshapeReshape0dense_features/userGenre5_indicator/Sum:output:0:dense_features/userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????x
-dense_features/userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)dense_features/userRatingCount/ExpandDims
ExpandDimsinputs_userratingcount6dense_features/userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
#dense_features/userRatingCount/CastCast2dense_features/userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????{
$dense_features/userRatingCount/ShapeShape'dense_features/userRatingCount/Cast:y:0*
T0*
_output_shapes
:|
2dense_features/userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,dense_features/userRatingCount/strided_sliceStridedSlice-dense_features/userRatingCount/Shape:output:0;dense_features/userRatingCount/strided_slice/stack:output:0=dense_features/userRatingCount/strided_slice/stack_1:output:0=dense_features/userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/userRatingCount/Reshape/shapePack5dense_features/userRatingCount/strided_slice:output:07dense_features/userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&dense_features/userRatingCount/ReshapeReshape'dense_features/userRatingCount/Cast:y:05dense_features/userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
.dense_features/userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*dense_features/userRatingStddev/ExpandDims
ExpandDimsinputs_userratingstddev7dense_features/userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
%dense_features/userRatingStddev/ShapeShape3dense_features/userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:}
3dense_features/userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/userRatingStddev/strided_sliceStridedSlice.dense_features/userRatingStddev/Shape:output:0<dense_features/userRatingStddev/strided_slice/stack:output:0>dense_features/userRatingStddev/strided_slice/stack_1:output:0>dense_features/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/userRatingStddev/Reshape/shapePack6dense_features/userRatingStddev/strided_slice:output:08dense_features/userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/userRatingStddev/ReshapeReshape3dense_features/userRatingStddev/ExpandDims:output:06dense_features/userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2.dense_features/movieAvgRating/Reshape:output:05dense_features/movieGenre1_indicator/Reshape:output:05dense_features/movieGenre2_indicator/Reshape:output:05dense_features/movieGenre3_indicator/Reshape:output:00dense_features/movieRatingCount/Reshape:output:01dense_features/movieRatingStddev/Reshape:output:0+dense_features/releaseYear/Reshape:output:0-dense_features/userAvgRating/Reshape:output:04dense_features/userGenre1_indicator/Reshape:output:04dense_features/userGenre2_indicator/Reshape:output:04dense_features/userGenre3_indicator/Reshape:output:04dense_features/userGenre4_indicator/Reshape:output:04dense_features/userGenre5_indicator/Reshape:output:0/dense_features/userRatingCount/Reshape:output:00dense_features/userRatingStddev/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
fm_layer/MatMul/ReadVariableOpReadVariableOp'fm_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
fm_layer/MatMulMatMuldense_features/concat:output:0&fm_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
S
fm_layer/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
fm_layer/PowPowfm_layer/MatMul:product:0fm_layer/Pow/y:output:0*
T0*'
_output_shapes
:?????????
U
fm_layer/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
fm_layer/Pow_1Powdense_features/concat:output:0fm_layer/Pow_1/y:output:0*
T0*(
_output_shapes
:???????????
fm_layer/Pow_2/ReadVariableOpReadVariableOp'fm_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0U
fm_layer/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
fm_layer/Pow_2Pow%fm_layer/Pow_2/ReadVariableOp:value:0fm_layer/Pow_2/y:output:0*
T0*
_output_shapes
:	?
u
fm_layer/MatMul_1MatMulfm_layer/Pow_1:z:0fm_layer/Pow_2:z:0*
T0*'
_output_shapes
:?????????
t
fm_layer/subSubfm_layer/Pow:z:0fm_layer/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
`
fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
fm_layer/SumSumfm_layer/sub:z:0'fm_layer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(S
fm_layer/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
fm_layer/mulMulfm_layer/Sum:output:0fm_layer/mul/y:output:0*
T0*'
_output_shapes
:?????????l
add/addAddV2dense/BiasAdd:output:0fm_layer/mul:z:0*
T0*'
_output_shapes
:?????????\
activation/SigmoidSigmoidadd/add:z:0*
T0*'
_output_shapes
:??????????
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOpC^dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2C^dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2C^dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2^fm_layer/MatMul/ReadVariableOp^fm_layer/Pow_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2?
Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV22@
fm_layer/MatMul/ReadVariableOpfm_layer/MatMul/ReadVariableOp2>
fm_layer/Pow_2/ReadVariableOpfm_layer/Pow_2/ReadVariableOp:Z V
#
_output_shapes
:?????????
/
_user_specified_nameinputs/movieAvgRating:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre1:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre2:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre3:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/movieRatingCount:]Y
#
_output_shapes
:?????????
2
_user_specified_nameinputs/movieRatingStddev:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/releaseYear:YU
#
_output_shapes
:?????????
.
_user_specified_nameinputs/userAvgRating:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre1:V	R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre2:V
R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre3:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre4:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre5:[W
#
_output_shapes
:?????????
0
_user_specified_nameinputs/userRatingCount:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?:
?
@__inference_model_layer_call_and_return_conditional_losses_10146

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
dense_features_10090
dense_features_10092	
dense_features_10094
dense_features_10096	
dense_features_10098
dense_features_10100	
dense_features_10102
dense_features_10104	
dense_features_10106
dense_features_10108	
dense_features_10110
dense_features_10112	
dense_features_10114
dense_features_10116	
dense_features_10118
dense_features_10120	
dense_10123:	?
dense_10125:!
fm_layer_10128:	?

identity??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?&dense_features/StatefulPartitionedCall? fm_layer/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14dense_features_10090dense_features_10092dense_features_10094dense_features_10096dense_features_10098dense_features_10100dense_features_10102dense_features_10104dense_features_10106dense_features_10108dense_features_10110dense_features_10112dense_features_10114dense_features_10116dense_features_10118dense_features_10120**
Tin#
!2								*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_features_layer_call_and_return_conditional_losses_9965?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_10123dense_10125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_9466?
 fm_layer/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0fm_layer_10128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_fm_layer_layer_call_and_return_conditional_losses_9492?
add/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0)fm_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_9502?
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_9509x
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_10123*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10125*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp'^dense_features/StatefulPartitionedCall!^fm_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2D
 fm_layer/StatefulPartitionedCall fm_layer/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:K	G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:K
G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_121402
.table_init485_lookuptableimportv2_table_handle*
&table_init485_lookuptableimportv2_keys,
(table_init485_lookuptableimportv2_values	
identity??!table_init485/LookupTableImportV2?
!table_init485/LookupTableImportV2LookupTableImportV2.table_init485_lookuptableimportv2_table_handle&table_init485_lookuptableimportv2_keys(table_init485_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init485/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init485/LookupTableImportV2!table_init485/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
#__inference_signature_wrapper_10467
movieavgrating
moviegenre1
moviegenre2
moviegenre3
movieratingcount
movieratingstddev
releaseyear
useravgrating

usergenre1

usergenre2

usergenre3

usergenre4

usergenre5
userratingcount
userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15:	?

unknown_16:

unknown_17:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovieavgratingmoviegenre1moviegenre2moviegenre3movieratingcountmovieratingstddevreleaseyearuseravgrating
usergenre1
usergenre2
usergenre3
usergenre4
usergenre5userratingcountuserratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*-
Tin&
$2"								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
 !*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_9074o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namemovieAvgRating:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre1:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre2:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre3:UQ
#
_output_shapes
:?????????
*
_user_specified_namemovieRatingCount:VR
#
_output_shapes
:?????????
+
_user_specified_namemovieRatingStddev:PL
#
_output_shapes
:?????????
%
_user_specified_namereleaseYear:RN
#
_output_shapes
:?????????
'
_user_specified_nameuserAvgRating:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre1:O	K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre2:O
K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre3:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre4:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre5:TP
#
_output_shapes
:?????????
)
_user_specified_nameuserRatingCount:UQ
#
_output_shapes
:?????????
*
_user_specified_nameuserRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_122652
.table_init557_lookuptableimportv2_table_handle*
&table_init557_lookuptableimportv2_keys,
(table_init557_lookuptableimportv2_values	
identity??!table_init557/LookupTableImportV2?
!table_init557/LookupTableImportV2LookupTableImportV2.table_init557_lookuptableimportv2_table_handle&table_init557_lookuptableimportv2_keys(table_init557_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init557/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init557/LookupTableImportV2!table_init557/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?

?
__inference_loss_fn_1_12073C
5dense_bias_regularizer_square_readvariableop_resource:
identity??,dense/bias/Regularizer/Square/ReadVariableOp?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: \
IdentityIdentitydense/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^dense/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp
?
,
__inference__destroyer_12199
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
%__inference_model_layer_call_fn_10244
movieavgrating
moviegenre1
moviegenre2
moviegenre3
movieratingcount
movieratingstddev
releaseyear
useravgrating

usergenre1

usergenre2

usergenre3

usergenre4

usergenre5
userratingcount
userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15:	?

unknown_16:

unknown_17:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovieavgratingmoviegenre1moviegenre2moviegenre3movieratingcountmovieratingstddevreleaseyearuseravgrating
usergenre1
usergenre2
usergenre3
usergenre4
usergenre5userratingcountuserratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*-
Tin&
$2"								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
 !*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namemovieAvgRating:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre1:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre2:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre3:UQ
#
_output_shapes
:?????????
*
_user_specified_namemovieRatingCount:VR
#
_output_shapes
:?????????
+
_user_specified_namemovieRatingStddev:PL
#
_output_shapes
:?????????
%
_user_specified_namereleaseYear:RN
#
_output_shapes
:?????????
'
_user_specified_nameuserAvgRating:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre1:O	K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre2:O
K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre3:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre4:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre5:TP
#
_output_shapes
:?????????
)
_user_specified_nameuserRatingCount:UQ
#
_output_shapes
:?????????
*
_user_specified_nameuserRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
a
E__inference_activation_layer_call_and_return_conditional_losses_12051

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_121042
.table_init371_lookuptableimportv2_table_handle*
&table_init371_lookuptableimportv2_keys,
(table_init371_lookuptableimportv2_values	
identity??!table_init371/LookupTableImportV2?
!table_init371/LookupTableImportV2LookupTableImportV2.table_init371_lookuptableimportv2_table_handle&table_init371_lookuptableimportv2_keys(table_init371_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init371/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init371/LookupTableImportV2!table_init371/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference_<lambda>_122412
.table_init407_lookuptableimportv2_table_handle*
&table_init407_lookuptableimportv2_keys,
(table_init407_lookuptableimportv2_values	
identity??!table_init407/LookupTableImportV2?
!table_init407/LookupTableImportV2LookupTableImportV2.table_init407_lookuptableimportv2_table_handle&table_init407_lookuptableimportv2_keys(table_init407_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init407/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init407/LookupTableImportV2!table_init407/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
??
?
__inference__wrapped_model_9074
movieavgrating
moviegenre1
moviegenre2
moviegenre3
movieratingcount
movieratingstddev
releaseyear
useravgrating

usergenre1

usergenre2

usergenre3

usergenre4

usergenre5
userratingcount
userratingstddevY
Umodel_dense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleZ
Vmodel_dense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	Y
Umodel_dense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleZ
Vmodel_dense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	Y
Umodel_dense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleZ
Vmodel_dense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	X
Tmodel_dense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleY
Umodel_dense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	X
Tmodel_dense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleY
Umodel_dense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	X
Tmodel_dense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleY
Umodel_dense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	X
Tmodel_dense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleY
Umodel_dense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	X
Tmodel_dense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleY
Umodel_dense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	=
*model_dense_matmul_readvariableop_resource:	?9
+model_dense_biasadd_readvariableop_resource:@
-model_fm_layer_matmul_readvariableop_resource:	?

identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?Hmodel/dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2?Hmodel/dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2?Hmodel/dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2?Gmodel/dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2?Gmodel/dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2?Gmodel/dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2?Gmodel/dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2?Gmodel/dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2?$model/fm_layer/MatMul/ReadVariableOp?#model/fm_layer/Pow_2/ReadVariableOp}
2model/dense_features/movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.model/dense_features/movieAvgRating/ExpandDims
ExpandDimsmovieavgrating;model/dense_features/movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
)model/dense_features/movieAvgRating/ShapeShape7model/dense_features/movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:?
7model/dense_features/movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9model/dense_features/movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9model/dense_features/movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1model/dense_features/movieAvgRating/strided_sliceStridedSlice2model/dense_features/movieAvgRating/Shape:output:0@model/dense_features/movieAvgRating/strided_slice/stack:output:0Bmodel/dense_features/movieAvgRating/strided_slice/stack_1:output:0Bmodel/dense_features/movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3model/dense_features/movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1model/dense_features/movieAvgRating/Reshape/shapePack:model/dense_features/movieAvgRating/strided_slice:output:0<model/dense_features/movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+model/dense_features/movieAvgRating/ReshapeReshape7model/dense_features/movieAvgRating/ExpandDims:output:0:model/dense_features/movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
9model/dense_features/movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5model/dense_features/movieGenre1_indicator/ExpandDims
ExpandDimsmoviegenre1Bmodel/dense_features/movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Imodel/dense_features/movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Cmodel/dense_features/movieGenre1_indicator/to_sparse_input/NotEqualNotEqual>model/dense_features/movieGenre1_indicator/ExpandDims:output:0Rmodel/dense_features/movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Bmodel/dense_features/movieGenre1_indicator/to_sparse_input/indicesWhereGmodel/dense_features/movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Amodel/dense_features/movieGenre1_indicator/to_sparse_input/valuesGatherNd>model/dense_features/movieGenre1_indicator/ExpandDims:output:0Jmodel/dense_features/movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Fmodel/dense_features/movieGenre1_indicator/to_sparse_input/dense_shapeShape>model/dense_features/movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Hmodel/dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Umodel_dense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleJmodel/dense_features/movieGenre1_indicator/to_sparse_input/values:output:0Vmodel_dense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Fmodel/dense_features/movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
8model/dense_features/movieGenre1_indicator/SparseToDenseSparseToDenseJmodel/dense_features/movieGenre1_indicator/to_sparse_input/indices:index:0Omodel/dense_features/movieGenre1_indicator/to_sparse_input/dense_shape:output:0Qmodel/dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0Omodel/dense_features/movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????}
8model/dense_features/movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
:model/dense_features/movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    z
8model/dense_features/movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
2model/dense_features/movieGenre1_indicator/one_hotOneHot@model/dense_features/movieGenre1_indicator/SparseToDense:dense:0Amodel/dense_features/movieGenre1_indicator/one_hot/depth:output:0Amodel/dense_features/movieGenre1_indicator/one_hot/Const:output:0Cmodel/dense_features/movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
@model/dense_features/movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
.model/dense_features/movieGenre1_indicator/SumSum;model/dense_features/movieGenre1_indicator/one_hot:output:0Imodel/dense_features/movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
0model/dense_features/movieGenre1_indicator/ShapeShape7model/dense_features/movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:?
>model/dense_features/movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model/dense_features/movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model/dense_features/movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model/dense_features/movieGenre1_indicator/strided_sliceStridedSlice9model/dense_features/movieGenre1_indicator/Shape:output:0Gmodel/dense_features/movieGenre1_indicator/strided_slice/stack:output:0Imodel/dense_features/movieGenre1_indicator/strided_slice/stack_1:output:0Imodel/dense_features/movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:model/dense_features/movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8model/dense_features/movieGenre1_indicator/Reshape/shapePackAmodel/dense_features/movieGenre1_indicator/strided_slice:output:0Cmodel/dense_features/movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
2model/dense_features/movieGenre1_indicator/ReshapeReshape7model/dense_features/movieGenre1_indicator/Sum:output:0Amodel/dense_features/movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
9model/dense_features/movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5model/dense_features/movieGenre2_indicator/ExpandDims
ExpandDimsmoviegenre2Bmodel/dense_features/movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Imodel/dense_features/movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Cmodel/dense_features/movieGenre2_indicator/to_sparse_input/NotEqualNotEqual>model/dense_features/movieGenre2_indicator/ExpandDims:output:0Rmodel/dense_features/movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Bmodel/dense_features/movieGenre2_indicator/to_sparse_input/indicesWhereGmodel/dense_features/movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Amodel/dense_features/movieGenre2_indicator/to_sparse_input/valuesGatherNd>model/dense_features/movieGenre2_indicator/ExpandDims:output:0Jmodel/dense_features/movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Fmodel/dense_features/movieGenre2_indicator/to_sparse_input/dense_shapeShape>model/dense_features/movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Hmodel/dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Umodel_dense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleJmodel/dense_features/movieGenre2_indicator/to_sparse_input/values:output:0Vmodel_dense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Fmodel/dense_features/movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
8model/dense_features/movieGenre2_indicator/SparseToDenseSparseToDenseJmodel/dense_features/movieGenre2_indicator/to_sparse_input/indices:index:0Omodel/dense_features/movieGenre2_indicator/to_sparse_input/dense_shape:output:0Qmodel/dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0Omodel/dense_features/movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????}
8model/dense_features/movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
:model/dense_features/movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    z
8model/dense_features/movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
2model/dense_features/movieGenre2_indicator/one_hotOneHot@model/dense_features/movieGenre2_indicator/SparseToDense:dense:0Amodel/dense_features/movieGenre2_indicator/one_hot/depth:output:0Amodel/dense_features/movieGenre2_indicator/one_hot/Const:output:0Cmodel/dense_features/movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
@model/dense_features/movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
.model/dense_features/movieGenre2_indicator/SumSum;model/dense_features/movieGenre2_indicator/one_hot:output:0Imodel/dense_features/movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
0model/dense_features/movieGenre2_indicator/ShapeShape7model/dense_features/movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:?
>model/dense_features/movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model/dense_features/movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model/dense_features/movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model/dense_features/movieGenre2_indicator/strided_sliceStridedSlice9model/dense_features/movieGenre2_indicator/Shape:output:0Gmodel/dense_features/movieGenre2_indicator/strided_slice/stack:output:0Imodel/dense_features/movieGenre2_indicator/strided_slice/stack_1:output:0Imodel/dense_features/movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:model/dense_features/movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8model/dense_features/movieGenre2_indicator/Reshape/shapePackAmodel/dense_features/movieGenre2_indicator/strided_slice:output:0Cmodel/dense_features/movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
2model/dense_features/movieGenre2_indicator/ReshapeReshape7model/dense_features/movieGenre2_indicator/Sum:output:0Amodel/dense_features/movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
9model/dense_features/movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5model/dense_features/movieGenre3_indicator/ExpandDims
ExpandDimsmoviegenre3Bmodel/dense_features/movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Imodel/dense_features/movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Cmodel/dense_features/movieGenre3_indicator/to_sparse_input/NotEqualNotEqual>model/dense_features/movieGenre3_indicator/ExpandDims:output:0Rmodel/dense_features/movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Bmodel/dense_features/movieGenre3_indicator/to_sparse_input/indicesWhereGmodel/dense_features/movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Amodel/dense_features/movieGenre3_indicator/to_sparse_input/valuesGatherNd>model/dense_features/movieGenre3_indicator/ExpandDims:output:0Jmodel/dense_features/movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Fmodel/dense_features/movieGenre3_indicator/to_sparse_input/dense_shapeShape>model/dense_features/movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Hmodel/dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Umodel_dense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleJmodel/dense_features/movieGenre3_indicator/to_sparse_input/values:output:0Vmodel_dense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Fmodel/dense_features/movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
8model/dense_features/movieGenre3_indicator/SparseToDenseSparseToDenseJmodel/dense_features/movieGenre3_indicator/to_sparse_input/indices:index:0Omodel/dense_features/movieGenre3_indicator/to_sparse_input/dense_shape:output:0Qmodel/dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0Omodel/dense_features/movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????}
8model/dense_features/movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
:model/dense_features/movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    z
8model/dense_features/movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
2model/dense_features/movieGenre3_indicator/one_hotOneHot@model/dense_features/movieGenre3_indicator/SparseToDense:dense:0Amodel/dense_features/movieGenre3_indicator/one_hot/depth:output:0Amodel/dense_features/movieGenre3_indicator/one_hot/Const:output:0Cmodel/dense_features/movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
@model/dense_features/movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
.model/dense_features/movieGenre3_indicator/SumSum;model/dense_features/movieGenre3_indicator/one_hot:output:0Imodel/dense_features/movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
0model/dense_features/movieGenre3_indicator/ShapeShape7model/dense_features/movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:?
>model/dense_features/movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model/dense_features/movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model/dense_features/movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model/dense_features/movieGenre3_indicator/strided_sliceStridedSlice9model/dense_features/movieGenre3_indicator/Shape:output:0Gmodel/dense_features/movieGenre3_indicator/strided_slice/stack:output:0Imodel/dense_features/movieGenre3_indicator/strided_slice/stack_1:output:0Imodel/dense_features/movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:model/dense_features/movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8model/dense_features/movieGenre3_indicator/Reshape/shapePackAmodel/dense_features/movieGenre3_indicator/strided_slice:output:0Cmodel/dense_features/movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
2model/dense_features/movieGenre3_indicator/ReshapeReshape7model/dense_features/movieGenre3_indicator/Sum:output:0Amodel/dense_features/movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4model/dense_features/movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0model/dense_features/movieRatingCount/ExpandDims
ExpandDimsmovieratingcount=model/dense_features/movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
*model/dense_features/movieRatingCount/CastCast9model/dense_features/movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:??????????
+model/dense_features/movieRatingCount/ShapeShape.model/dense_features/movieRatingCount/Cast:y:0*
T0*
_output_shapes
:?
9model/dense_features/movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;model/dense_features/movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;model/dense_features/movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3model/dense_features/movieRatingCount/strided_sliceStridedSlice4model/dense_features/movieRatingCount/Shape:output:0Bmodel/dense_features/movieRatingCount/strided_slice/stack:output:0Dmodel/dense_features/movieRatingCount/strided_slice/stack_1:output:0Dmodel/dense_features/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5model/dense_features/movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
3model/dense_features/movieRatingCount/Reshape/shapePack<model/dense_features/movieRatingCount/strided_slice:output:0>model/dense_features/movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
-model/dense_features/movieRatingCount/ReshapeReshape.model/dense_features/movieRatingCount/Cast:y:0<model/dense_features/movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
5model/dense_features/movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1model/dense_features/movieRatingStddev/ExpandDims
ExpandDimsmovieratingstddev>model/dense_features/movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
,model/dense_features/movieRatingStddev/ShapeShape:model/dense_features/movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:?
:model/dense_features/movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<model/dense_features/movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<model/dense_features/movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4model/dense_features/movieRatingStddev/strided_sliceStridedSlice5model/dense_features/movieRatingStddev/Shape:output:0Cmodel/dense_features/movieRatingStddev/strided_slice/stack:output:0Emodel/dense_features/movieRatingStddev/strided_slice/stack_1:output:0Emodel/dense_features/movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6model/dense_features/movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4model/dense_features/movieRatingStddev/Reshape/shapePack=model/dense_features/movieRatingStddev/strided_slice:output:0?model/dense_features/movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.model/dense_features/movieRatingStddev/ReshapeReshape:model/dense_features/movieRatingStddev/ExpandDims:output:0=model/dense_features/movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/model/dense_features/releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+model/dense_features/releaseYear/ExpandDims
ExpandDimsreleaseyear8model/dense_features/releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
%model/dense_features/releaseYear/CastCast4model/dense_features/releaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????
&model/dense_features/releaseYear/ShapeShape)model/dense_features/releaseYear/Cast:y:0*
T0*
_output_shapes
:~
4model/dense_features/releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6model/dense_features/releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model/dense_features/releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model/dense_features/releaseYear/strided_sliceStridedSlice/model/dense_features/releaseYear/Shape:output:0=model/dense_features/releaseYear/strided_slice/stack:output:0?model/dense_features/releaseYear/strided_slice/stack_1:output:0?model/dense_features/releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0model/dense_features/releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.model/dense_features/releaseYear/Reshape/shapePack7model/dense_features/releaseYear/strided_slice:output:09model/dense_features/releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(model/dense_features/releaseYear/ReshapeReshape)model/dense_features/releaseYear/Cast:y:07model/dense_features/releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
1model/dense_features/userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-model/dense_features/userAvgRating/ExpandDims
ExpandDimsuseravgrating:model/dense_features/userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
(model/dense_features/userAvgRating/ShapeShape6model/dense_features/userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:?
6model/dense_features/userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8model/dense_features/userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8model/dense_features/userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model/dense_features/userAvgRating/strided_sliceStridedSlice1model/dense_features/userAvgRating/Shape:output:0?model/dense_features/userAvgRating/strided_slice/stack:output:0Amodel/dense_features/userAvgRating/strided_slice/stack_1:output:0Amodel/dense_features/userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2model/dense_features/userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
0model/dense_features/userAvgRating/Reshape/shapePack9model/dense_features/userAvgRating/strided_slice:output:0;model/dense_features/userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
*model/dense_features/userAvgRating/ReshapeReshape6model/dense_features/userAvgRating/ExpandDims:output:09model/dense_features/userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
8model/dense_features/userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4model/dense_features/userGenre1_indicator/ExpandDims
ExpandDims
usergenre1Amodel/dense_features/userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Hmodel/dense_features/userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Bmodel/dense_features/userGenre1_indicator/to_sparse_input/NotEqualNotEqual=model/dense_features/userGenre1_indicator/ExpandDims:output:0Qmodel/dense_features/userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Amodel/dense_features/userGenre1_indicator/to_sparse_input/indicesWhereFmodel/dense_features/userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
@model/dense_features/userGenre1_indicator/to_sparse_input/valuesGatherNd=model/dense_features/userGenre1_indicator/ExpandDims:output:0Imodel/dense_features/userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Emodel/dense_features/userGenre1_indicator/to_sparse_input/dense_shapeShape=model/dense_features/userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gmodel/dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Tmodel_dense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleImodel/dense_features/userGenre1_indicator/to_sparse_input/values:output:0Umodel_dense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Emodel/dense_features/userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
7model/dense_features/userGenre1_indicator/SparseToDenseSparseToDenseImodel/dense_features/userGenre1_indicator/to_sparse_input/indices:index:0Nmodel/dense_features/userGenre1_indicator/to_sparse_input/dense_shape:output:0Pmodel/dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2:values:0Nmodel/dense_features/userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????|
7model/dense_features/userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
9model/dense_features/userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    y
7model/dense_features/userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
1model/dense_features/userGenre1_indicator/one_hotOneHot?model/dense_features/userGenre1_indicator/SparseToDense:dense:0@model/dense_features/userGenre1_indicator/one_hot/depth:output:0@model/dense_features/userGenre1_indicator/one_hot/Const:output:0Bmodel/dense_features/userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
?model/dense_features/userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
-model/dense_features/userGenre1_indicator/SumSum:model/dense_features/userGenre1_indicator/one_hot:output:0Hmodel/dense_features/userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
/model/dense_features/userGenre1_indicator/ShapeShape6model/dense_features/userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:?
=model/dense_features/userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?model/dense_features/userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?model/dense_features/userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model/dense_features/userGenre1_indicator/strided_sliceStridedSlice8model/dense_features/userGenre1_indicator/Shape:output:0Fmodel/dense_features/userGenre1_indicator/strided_slice/stack:output:0Hmodel/dense_features/userGenre1_indicator/strided_slice/stack_1:output:0Hmodel/dense_features/userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9model/dense_features/userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
7model/dense_features/userGenre1_indicator/Reshape/shapePack@model/dense_features/userGenre1_indicator/strided_slice:output:0Bmodel/dense_features/userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
1model/dense_features/userGenre1_indicator/ReshapeReshape6model/dense_features/userGenre1_indicator/Sum:output:0@model/dense_features/userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
8model/dense_features/userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4model/dense_features/userGenre2_indicator/ExpandDims
ExpandDims
usergenre2Amodel/dense_features/userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Hmodel/dense_features/userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Bmodel/dense_features/userGenre2_indicator/to_sparse_input/NotEqualNotEqual=model/dense_features/userGenre2_indicator/ExpandDims:output:0Qmodel/dense_features/userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Amodel/dense_features/userGenre2_indicator/to_sparse_input/indicesWhereFmodel/dense_features/userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
@model/dense_features/userGenre2_indicator/to_sparse_input/valuesGatherNd=model/dense_features/userGenre2_indicator/ExpandDims:output:0Imodel/dense_features/userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Emodel/dense_features/userGenre2_indicator/to_sparse_input/dense_shapeShape=model/dense_features/userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gmodel/dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Tmodel_dense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleImodel/dense_features/userGenre2_indicator/to_sparse_input/values:output:0Umodel_dense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Emodel/dense_features/userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
7model/dense_features/userGenre2_indicator/SparseToDenseSparseToDenseImodel/dense_features/userGenre2_indicator/to_sparse_input/indices:index:0Nmodel/dense_features/userGenre2_indicator/to_sparse_input/dense_shape:output:0Pmodel/dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2:values:0Nmodel/dense_features/userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????|
7model/dense_features/userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
9model/dense_features/userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    y
7model/dense_features/userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
1model/dense_features/userGenre2_indicator/one_hotOneHot?model/dense_features/userGenre2_indicator/SparseToDense:dense:0@model/dense_features/userGenre2_indicator/one_hot/depth:output:0@model/dense_features/userGenre2_indicator/one_hot/Const:output:0Bmodel/dense_features/userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
?model/dense_features/userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
-model/dense_features/userGenre2_indicator/SumSum:model/dense_features/userGenre2_indicator/one_hot:output:0Hmodel/dense_features/userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
/model/dense_features/userGenre2_indicator/ShapeShape6model/dense_features/userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:?
=model/dense_features/userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?model/dense_features/userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?model/dense_features/userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model/dense_features/userGenre2_indicator/strided_sliceStridedSlice8model/dense_features/userGenre2_indicator/Shape:output:0Fmodel/dense_features/userGenre2_indicator/strided_slice/stack:output:0Hmodel/dense_features/userGenre2_indicator/strided_slice/stack_1:output:0Hmodel/dense_features/userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9model/dense_features/userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
7model/dense_features/userGenre2_indicator/Reshape/shapePack@model/dense_features/userGenre2_indicator/strided_slice:output:0Bmodel/dense_features/userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
1model/dense_features/userGenre2_indicator/ReshapeReshape6model/dense_features/userGenre2_indicator/Sum:output:0@model/dense_features/userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
8model/dense_features/userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4model/dense_features/userGenre3_indicator/ExpandDims
ExpandDims
usergenre3Amodel/dense_features/userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Hmodel/dense_features/userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Bmodel/dense_features/userGenre3_indicator/to_sparse_input/NotEqualNotEqual=model/dense_features/userGenre3_indicator/ExpandDims:output:0Qmodel/dense_features/userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Amodel/dense_features/userGenre3_indicator/to_sparse_input/indicesWhereFmodel/dense_features/userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
@model/dense_features/userGenre3_indicator/to_sparse_input/valuesGatherNd=model/dense_features/userGenre3_indicator/ExpandDims:output:0Imodel/dense_features/userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Emodel/dense_features/userGenre3_indicator/to_sparse_input/dense_shapeShape=model/dense_features/userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gmodel/dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Tmodel_dense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleImodel/dense_features/userGenre3_indicator/to_sparse_input/values:output:0Umodel_dense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Emodel/dense_features/userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
7model/dense_features/userGenre3_indicator/SparseToDenseSparseToDenseImodel/dense_features/userGenre3_indicator/to_sparse_input/indices:index:0Nmodel/dense_features/userGenre3_indicator/to_sparse_input/dense_shape:output:0Pmodel/dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2:values:0Nmodel/dense_features/userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????|
7model/dense_features/userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
9model/dense_features/userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    y
7model/dense_features/userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
1model/dense_features/userGenre3_indicator/one_hotOneHot?model/dense_features/userGenre3_indicator/SparseToDense:dense:0@model/dense_features/userGenre3_indicator/one_hot/depth:output:0@model/dense_features/userGenre3_indicator/one_hot/Const:output:0Bmodel/dense_features/userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
?model/dense_features/userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
-model/dense_features/userGenre3_indicator/SumSum:model/dense_features/userGenre3_indicator/one_hot:output:0Hmodel/dense_features/userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
/model/dense_features/userGenre3_indicator/ShapeShape6model/dense_features/userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:?
=model/dense_features/userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?model/dense_features/userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?model/dense_features/userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model/dense_features/userGenre3_indicator/strided_sliceStridedSlice8model/dense_features/userGenre3_indicator/Shape:output:0Fmodel/dense_features/userGenre3_indicator/strided_slice/stack:output:0Hmodel/dense_features/userGenre3_indicator/strided_slice/stack_1:output:0Hmodel/dense_features/userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9model/dense_features/userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
7model/dense_features/userGenre3_indicator/Reshape/shapePack@model/dense_features/userGenre3_indicator/strided_slice:output:0Bmodel/dense_features/userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
1model/dense_features/userGenre3_indicator/ReshapeReshape6model/dense_features/userGenre3_indicator/Sum:output:0@model/dense_features/userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
8model/dense_features/userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4model/dense_features/userGenre4_indicator/ExpandDims
ExpandDims
usergenre4Amodel/dense_features/userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Hmodel/dense_features/userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Bmodel/dense_features/userGenre4_indicator/to_sparse_input/NotEqualNotEqual=model/dense_features/userGenre4_indicator/ExpandDims:output:0Qmodel/dense_features/userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Amodel/dense_features/userGenre4_indicator/to_sparse_input/indicesWhereFmodel/dense_features/userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
@model/dense_features/userGenre4_indicator/to_sparse_input/valuesGatherNd=model/dense_features/userGenre4_indicator/ExpandDims:output:0Imodel/dense_features/userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Emodel/dense_features/userGenre4_indicator/to_sparse_input/dense_shapeShape=model/dense_features/userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gmodel/dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Tmodel_dense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleImodel/dense_features/userGenre4_indicator/to_sparse_input/values:output:0Umodel_dense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Emodel/dense_features/userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
7model/dense_features/userGenre4_indicator/SparseToDenseSparseToDenseImodel/dense_features/userGenre4_indicator/to_sparse_input/indices:index:0Nmodel/dense_features/userGenre4_indicator/to_sparse_input/dense_shape:output:0Pmodel/dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2:values:0Nmodel/dense_features/userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????|
7model/dense_features/userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
9model/dense_features/userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    y
7model/dense_features/userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
1model/dense_features/userGenre4_indicator/one_hotOneHot?model/dense_features/userGenre4_indicator/SparseToDense:dense:0@model/dense_features/userGenre4_indicator/one_hot/depth:output:0@model/dense_features/userGenre4_indicator/one_hot/Const:output:0Bmodel/dense_features/userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
?model/dense_features/userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
-model/dense_features/userGenre4_indicator/SumSum:model/dense_features/userGenre4_indicator/one_hot:output:0Hmodel/dense_features/userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
/model/dense_features/userGenre4_indicator/ShapeShape6model/dense_features/userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:?
=model/dense_features/userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?model/dense_features/userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?model/dense_features/userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model/dense_features/userGenre4_indicator/strided_sliceStridedSlice8model/dense_features/userGenre4_indicator/Shape:output:0Fmodel/dense_features/userGenre4_indicator/strided_slice/stack:output:0Hmodel/dense_features/userGenre4_indicator/strided_slice/stack_1:output:0Hmodel/dense_features/userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9model/dense_features/userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
7model/dense_features/userGenre4_indicator/Reshape/shapePack@model/dense_features/userGenre4_indicator/strided_slice:output:0Bmodel/dense_features/userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
1model/dense_features/userGenre4_indicator/ReshapeReshape6model/dense_features/userGenre4_indicator/Sum:output:0@model/dense_features/userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
8model/dense_features/userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4model/dense_features/userGenre5_indicator/ExpandDims
ExpandDims
usergenre5Amodel/dense_features/userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Hmodel/dense_features/userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Bmodel/dense_features/userGenre5_indicator/to_sparse_input/NotEqualNotEqual=model/dense_features/userGenre5_indicator/ExpandDims:output:0Qmodel/dense_features/userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Amodel/dense_features/userGenre5_indicator/to_sparse_input/indicesWhereFmodel/dense_features/userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
@model/dense_features/userGenre5_indicator/to_sparse_input/valuesGatherNd=model/dense_features/userGenre5_indicator/ExpandDims:output:0Imodel/dense_features/userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Emodel/dense_features/userGenre5_indicator/to_sparse_input/dense_shapeShape=model/dense_features/userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Gmodel/dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Tmodel_dense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleImodel/dense_features/userGenre5_indicator/to_sparse_input/values:output:0Umodel_dense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Emodel/dense_features/userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
7model/dense_features/userGenre5_indicator/SparseToDenseSparseToDenseImodel/dense_features/userGenre5_indicator/to_sparse_input/indices:index:0Nmodel/dense_features/userGenre5_indicator/to_sparse_input/dense_shape:output:0Pmodel/dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2:values:0Nmodel/dense_features/userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????|
7model/dense_features/userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
9model/dense_features/userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    y
7model/dense_features/userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
1model/dense_features/userGenre5_indicator/one_hotOneHot?model/dense_features/userGenre5_indicator/SparseToDense:dense:0@model/dense_features/userGenre5_indicator/one_hot/depth:output:0@model/dense_features/userGenre5_indicator/one_hot/Const:output:0Bmodel/dense_features/userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
?model/dense_features/userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
-model/dense_features/userGenre5_indicator/SumSum:model/dense_features/userGenre5_indicator/one_hot:output:0Hmodel/dense_features/userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
/model/dense_features/userGenre5_indicator/ShapeShape6model/dense_features/userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:?
=model/dense_features/userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?model/dense_features/userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?model/dense_features/userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model/dense_features/userGenre5_indicator/strided_sliceStridedSlice8model/dense_features/userGenre5_indicator/Shape:output:0Fmodel/dense_features/userGenre5_indicator/strided_slice/stack:output:0Hmodel/dense_features/userGenre5_indicator/strided_slice/stack_1:output:0Hmodel/dense_features/userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9model/dense_features/userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
7model/dense_features/userGenre5_indicator/Reshape/shapePack@model/dense_features/userGenre5_indicator/strided_slice:output:0Bmodel/dense_features/userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
1model/dense_features/userGenre5_indicator/ReshapeReshape6model/dense_features/userGenre5_indicator/Sum:output:0@model/dense_features/userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3model/dense_features/userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/model/dense_features/userRatingCount/ExpandDims
ExpandDimsuserratingcount<model/dense_features/userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
)model/dense_features/userRatingCount/CastCast8model/dense_features/userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:??????????
*model/dense_features/userRatingCount/ShapeShape-model/dense_features/userRatingCount/Cast:y:0*
T0*
_output_shapes
:?
8model/dense_features/userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:model/dense_features/userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:model/dense_features/userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2model/dense_features/userRatingCount/strided_sliceStridedSlice3model/dense_features/userRatingCount/Shape:output:0Amodel/dense_features/userRatingCount/strided_slice/stack:output:0Cmodel/dense_features/userRatingCount/strided_slice/stack_1:output:0Cmodel/dense_features/userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4model/dense_features/userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2model/dense_features/userRatingCount/Reshape/shapePack;model/dense_features/userRatingCount/strided_slice:output:0=model/dense_features/userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,model/dense_features/userRatingCount/ReshapeReshape-model/dense_features/userRatingCount/Cast:y:0;model/dense_features/userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4model/dense_features/userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0model/dense_features/userRatingStddev/ExpandDims
ExpandDimsuserratingstddev=model/dense_features/userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
+model/dense_features/userRatingStddev/ShapeShape9model/dense_features/userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:?
9model/dense_features/userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;model/dense_features/userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;model/dense_features/userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3model/dense_features/userRatingStddev/strided_sliceStridedSlice4model/dense_features/userRatingStddev/Shape:output:0Bmodel/dense_features/userRatingStddev/strided_slice/stack:output:0Dmodel/dense_features/userRatingStddev/strided_slice/stack_1:output:0Dmodel/dense_features/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5model/dense_features/userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
3model/dense_features/userRatingStddev/Reshape/shapePack<model/dense_features/userRatingStddev/strided_slice:output:0>model/dense_features/userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
-model/dense_features/userRatingStddev/ReshapeReshape9model/dense_features/userRatingStddev/ExpandDims:output:0<model/dense_features/userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 model/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
model/dense_features/concatConcatV24model/dense_features/movieAvgRating/Reshape:output:0;model/dense_features/movieGenre1_indicator/Reshape:output:0;model/dense_features/movieGenre2_indicator/Reshape:output:0;model/dense_features/movieGenre3_indicator/Reshape:output:06model/dense_features/movieRatingCount/Reshape:output:07model/dense_features/movieRatingStddev/Reshape:output:01model/dense_features/releaseYear/Reshape:output:03model/dense_features/userAvgRating/Reshape:output:0:model/dense_features/userGenre1_indicator/Reshape:output:0:model/dense_features/userGenre2_indicator/Reshape:output:0:model/dense_features/userGenre3_indicator/Reshape:output:0:model/dense_features/userGenre4_indicator/Reshape:output:0:model/dense_features/userGenre5_indicator/Reshape:output:05model/dense_features/userRatingCount/Reshape:output:06model/dense_features/userRatingStddev/Reshape:output:0)model/dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model/dense/MatMulMatMul$model/dense_features/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/fm_layer/MatMul/ReadVariableOpReadVariableOp-model_fm_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
model/fm_layer/MatMulMatMul$model/dense_features/concat:output:0,model/fm_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
Y
model/fm_layer/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
model/fm_layer/PowPowmodel/fm_layer/MatMul:product:0model/fm_layer/Pow/y:output:0*
T0*'
_output_shapes
:?????????
[
model/fm_layer/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
model/fm_layer/Pow_1Pow$model/dense_features/concat:output:0model/fm_layer/Pow_1/y:output:0*
T0*(
_output_shapes
:???????????
#model/fm_layer/Pow_2/ReadVariableOpReadVariableOp-model_fm_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0[
model/fm_layer/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
model/fm_layer/Pow_2Pow+model/fm_layer/Pow_2/ReadVariableOp:value:0model/fm_layer/Pow_2/y:output:0*
T0*
_output_shapes
:	?
?
model/fm_layer/MatMul_1MatMulmodel/fm_layer/Pow_1:z:0model/fm_layer/Pow_2:z:0*
T0*'
_output_shapes
:?????????
?
model/fm_layer/subSubmodel/fm_layer/Pow:z:0!model/fm_layer/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
f
$model/fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
model/fm_layer/SumSummodel/fm_layer/sub:z:0-model/fm_layer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(Y
model/fm_layer/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
model/fm_layer/mulMulmodel/fm_layer/Sum:output:0model/fm_layer/mul/y:output:0*
T0*'
_output_shapes
:?????????~
model/add/addAddV2model/dense/BiasAdd:output:0model/fm_layer/mul:z:0*
T0*'
_output_shapes
:?????????h
model/activation/SigmoidSigmoidmodel/add/add:z:0*
T0*'
_output_shapes
:?????????k
IdentityIdentitymodel/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOpI^model/dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2I^model/dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2I^model/dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2H^model/dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2H^model/dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2H^model/dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2H^model/dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2H^model/dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2%^model/fm_layer/MatMul/ReadVariableOp$^model/fm_layer/Pow_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2?
Hmodel/dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2Hmodel/dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV22?
Hmodel/dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2Hmodel/dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV22?
Hmodel/dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2Hmodel/dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV22?
Gmodel/dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2Gmodel/dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV22?
Gmodel/dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2Gmodel/dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV22?
Gmodel/dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2Gmodel/dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV22?
Gmodel/dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2Gmodel/dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV22?
Gmodel/dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2Gmodel/dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV22L
$model/fm_layer/MatMul/ReadVariableOp$model/fm_layer/MatMul/ReadVariableOp2J
#model/fm_layer/Pow_2/ReadVariableOp#model/fm_layer/Pow_2/ReadVariableOp:S O
#
_output_shapes
:?????????
(
_user_specified_namemovieAvgRating:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre1:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre2:PL
#
_output_shapes
:?????????
%
_user_specified_namemovieGenre3:UQ
#
_output_shapes
:?????????
*
_user_specified_namemovieRatingCount:VR
#
_output_shapes
:?????????
+
_user_specified_namemovieRatingStddev:PL
#
_output_shapes
:?????????
%
_user_specified_namereleaseYear:RN
#
_output_shapes
:?????????
'
_user_specified_nameuserAvgRating:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre1:O	K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre2:O
K
#
_output_shapes
:?????????
$
_user_specified_name
userGenre3:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre4:OK
#
_output_shapes
:?????????
$
_user_specified_name
userGenre5:TP
#
_output_shapes
:?????????
)
_user_specified_nameuserRatingCount:UQ
#
_output_shapes
:?????????
*
_user_specified_nameuserRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
@__inference_model_layer_call_and_return_conditional_losses_11267
inputs_movieavgrating
inputs_moviegenre1
inputs_moviegenre2
inputs_moviegenre3
inputs_movieratingcount
inputs_movieratingstddev
inputs_releaseyear
inputs_useravgrating
inputs_usergenre1
inputs_usergenre2
inputs_usergenre3
inputs_usergenre4
inputs_usergenre5
inputs_userratingcount
inputs_userratingstddevS
Odense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource::
'fm_layer_matmul_readvariableop_resource:	?

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2?Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2?Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2?Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2?fm_layer/MatMul/ReadVariableOp?fm_layer/Pow_2/ReadVariableOpw
,dense_features/movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(dense_features/movieAvgRating/ExpandDims
ExpandDimsinputs_movieavgrating5dense_features/movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
#dense_features/movieAvgRating/ShapeShape1dense_features/movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:{
1dense_features/movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_features/movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_features/movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_features/movieAvgRating/strided_sliceStridedSlice,dense_features/movieAvgRating/Shape:output:0:dense_features/movieAvgRating/strided_slice/stack:output:0<dense_features/movieAvgRating/strided_slice/stack_1:output:0<dense_features/movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-dense_features/movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/movieAvgRating/Reshape/shapePack4dense_features/movieAvgRating/strided_slice:output:06dense_features/movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%dense_features/movieAvgRating/ReshapeReshape1dense_features/movieAvgRating/ExpandDims:output:04dense_features/movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/movieGenre1_indicator/ExpandDims
ExpandDimsinputs_moviegenre1<dense_features/movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/movieGenre1_indicator/to_sparse_input/NotEqualNotEqual8dense_features/movieGenre1_indicator/ExpandDims:output:0Ldense_features/movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/movieGenre1_indicator/to_sparse_input/indicesWhereAdense_features/movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/movieGenre1_indicator/to_sparse_input/valuesGatherNd8dense_features/movieGenre1_indicator/ExpandDims:output:0Ddense_features/movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/movieGenre1_indicator/to_sparse_input/dense_shapeShape8dense_features/movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/movieGenre1_indicator/to_sparse_input/values:output:0Pdense_features_moviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/movieGenre1_indicator/SparseToDenseSparseToDenseDdense_features/movieGenre1_indicator/to_sparse_input/indices:index:0Idense_features/movieGenre1_indicator/to_sparse_input/dense_shape:output:0Kdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/movieGenre1_indicator/one_hotOneHot:dense_features/movieGenre1_indicator/SparseToDense:dense:0;dense_features/movieGenre1_indicator/one_hot/depth:output:0;dense_features/movieGenre1_indicator/one_hot/Const:output:0=dense_features/movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/movieGenre1_indicator/SumSum5dense_features/movieGenre1_indicator/one_hot:output:0Cdense_features/movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/movieGenre1_indicator/ShapeShape1dense_features/movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/movieGenre1_indicator/strided_sliceStridedSlice3dense_features/movieGenre1_indicator/Shape:output:0Adense_features/movieGenre1_indicator/strided_slice/stack:output:0Cdense_features/movieGenre1_indicator/strided_slice/stack_1:output:0Cdense_features/movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/movieGenre1_indicator/Reshape/shapePack;dense_features/movieGenre1_indicator/strided_slice:output:0=dense_features/movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/movieGenre1_indicator/ReshapeReshape1dense_features/movieGenre1_indicator/Sum:output:0;dense_features/movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/movieGenre2_indicator/ExpandDims
ExpandDimsinputs_moviegenre2<dense_features/movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/movieGenre2_indicator/to_sparse_input/NotEqualNotEqual8dense_features/movieGenre2_indicator/ExpandDims:output:0Ldense_features/movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/movieGenre2_indicator/to_sparse_input/indicesWhereAdense_features/movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/movieGenre2_indicator/to_sparse_input/valuesGatherNd8dense_features/movieGenre2_indicator/ExpandDims:output:0Ddense_features/movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/movieGenre2_indicator/to_sparse_input/dense_shapeShape8dense_features/movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/movieGenre2_indicator/to_sparse_input/values:output:0Pdense_features_moviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/movieGenre2_indicator/SparseToDenseSparseToDenseDdense_features/movieGenre2_indicator/to_sparse_input/indices:index:0Idense_features/movieGenre2_indicator/to_sparse_input/dense_shape:output:0Kdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/movieGenre2_indicator/one_hotOneHot:dense_features/movieGenre2_indicator/SparseToDense:dense:0;dense_features/movieGenre2_indicator/one_hot/depth:output:0;dense_features/movieGenre2_indicator/one_hot/Const:output:0=dense_features/movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/movieGenre2_indicator/SumSum5dense_features/movieGenre2_indicator/one_hot:output:0Cdense_features/movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/movieGenre2_indicator/ShapeShape1dense_features/movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/movieGenre2_indicator/strided_sliceStridedSlice3dense_features/movieGenre2_indicator/Shape:output:0Adense_features/movieGenre2_indicator/strided_slice/stack:output:0Cdense_features/movieGenre2_indicator/strided_slice/stack_1:output:0Cdense_features/movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/movieGenre2_indicator/Reshape/shapePack;dense_features/movieGenre2_indicator/strided_slice:output:0=dense_features/movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/movieGenre2_indicator/ReshapeReshape1dense_features/movieGenre2_indicator/Sum:output:0;dense_features/movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3dense_features/movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/dense_features/movieGenre3_indicator/ExpandDims
ExpandDimsinputs_moviegenre3<dense_features/movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/movieGenre3_indicator/to_sparse_input/NotEqualNotEqual8dense_features/movieGenre3_indicator/ExpandDims:output:0Ldense_features/movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/movieGenre3_indicator/to_sparse_input/indicesWhereAdense_features/movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/movieGenre3_indicator/to_sparse_input/valuesGatherNd8dense_features/movieGenre3_indicator/ExpandDims:output:0Ddense_features/movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/movieGenre3_indicator/to_sparse_input/dense_shapeShape8dense_features/movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/movieGenre3_indicator/to_sparse_input/values:output:0Pdense_features_moviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/movieGenre3_indicator/SparseToDenseSparseToDenseDdense_features/movieGenre3_indicator/to_sparse_input/indices:index:0Idense_features/movieGenre3_indicator/to_sparse_input/dense_shape:output:0Kdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/movieGenre3_indicator/one_hotOneHot:dense_features/movieGenre3_indicator/SparseToDense:dense:0;dense_features/movieGenre3_indicator/one_hot/depth:output:0;dense_features/movieGenre3_indicator/one_hot/Const:output:0=dense_features/movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/movieGenre3_indicator/SumSum5dense_features/movieGenre3_indicator/one_hot:output:0Cdense_features/movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/movieGenre3_indicator/ShapeShape1dense_features/movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/movieGenre3_indicator/strided_sliceStridedSlice3dense_features/movieGenre3_indicator/Shape:output:0Adense_features/movieGenre3_indicator/strided_slice/stack:output:0Cdense_features/movieGenre3_indicator/strided_slice/stack_1:output:0Cdense_features/movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/movieGenre3_indicator/Reshape/shapePack;dense_features/movieGenre3_indicator/strided_slice:output:0=dense_features/movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/movieGenre3_indicator/ReshapeReshape1dense_features/movieGenre3_indicator/Sum:output:0;dense_features/movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
.dense_features/movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*dense_features/movieRatingCount/ExpandDims
ExpandDimsinputs_movieratingcount7dense_features/movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
$dense_features/movieRatingCount/CastCast3dense_features/movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????}
%dense_features/movieRatingCount/ShapeShape(dense_features/movieRatingCount/Cast:y:0*
T0*
_output_shapes
:}
3dense_features/movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/movieRatingCount/strided_sliceStridedSlice.dense_features/movieRatingCount/Shape:output:0<dense_features/movieRatingCount/strided_slice/stack:output:0>dense_features/movieRatingCount/strided_slice/stack_1:output:0>dense_features/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/movieRatingCount/Reshape/shapePack6dense_features/movieRatingCount/strided_slice:output:08dense_features/movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/movieRatingCount/ReshapeReshape(dense_features/movieRatingCount/Cast:y:06dense_features/movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/dense_features/movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+dense_features/movieRatingStddev/ExpandDims
ExpandDimsinputs_movieratingstddev8dense_features/movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
&dense_features/movieRatingStddev/ShapeShape4dense_features/movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:~
4dense_features/movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/movieRatingStddev/strided_sliceStridedSlice/dense_features/movieRatingStddev/Shape:output:0=dense_features/movieRatingStddev/strided_slice/stack:output:0?dense_features/movieRatingStddev/strided_slice/stack_1:output:0?dense_features/movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/movieRatingStddev/Reshape/shapePack7dense_features/movieRatingStddev/strided_slice:output:09dense_features/movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/movieRatingStddev/ReshapeReshape4dense_features/movieRatingStddev/ExpandDims:output:07dense_features/movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
)dense_features/releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%dense_features/releaseYear/ExpandDims
ExpandDimsinputs_releaseyear2dense_features/releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
dense_features/releaseYear/CastCast.dense_features/releaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????s
 dense_features/releaseYear/ShapeShape#dense_features/releaseYear/Cast:y:0*
T0*
_output_shapes
:x
.dense_features/releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(dense_features/releaseYear/strided_sliceStridedSlice)dense_features/releaseYear/Shape:output:07dense_features/releaseYear/strided_slice/stack:output:09dense_features/releaseYear/strided_slice/stack_1:output:09dense_features/releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
(dense_features/releaseYear/Reshape/shapePack1dense_features/releaseYear/strided_slice:output:03dense_features/releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
"dense_features/releaseYear/ReshapeReshape#dense_features/releaseYear/Cast:y:01dense_features/releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????v
+dense_features/userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'dense_features/userAvgRating/ExpandDims
ExpandDimsinputs_useravgrating4dense_features/userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
"dense_features/userAvgRating/ShapeShape0dense_features/userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:z
0dense_features/userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_features/userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_features/userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*dense_features/userAvgRating/strided_sliceStridedSlice+dense_features/userAvgRating/Shape:output:09dense_features/userAvgRating/strided_slice/stack:output:0;dense_features/userAvgRating/strided_slice/stack_1:output:0;dense_features/userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dense_features/userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
*dense_features/userAvgRating/Reshape/shapePack3dense_features/userAvgRating/strided_slice:output:05dense_features/userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$dense_features/userAvgRating/ReshapeReshape0dense_features/userAvgRating/ExpandDims:output:03dense_features/userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre1_indicator/ExpandDims
ExpandDimsinputs_usergenre1;dense_features/userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre1_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre1_indicator/ExpandDims:output:0Kdense_features/userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre1_indicator/to_sparse_input/indicesWhere@dense_features/userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre1_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre1_indicator/ExpandDims:output:0Cdense_features/userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre1_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre1_indicator/to_sparse_input/values:output:0Odense_features_usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre1_indicator/SparseToDenseSparseToDenseCdense_features/userGenre1_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre1_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre1_indicator/one_hotOneHot9dense_features/userGenre1_indicator/SparseToDense:dense:0:dense_features/userGenre1_indicator/one_hot/depth:output:0:dense_features/userGenre1_indicator/one_hot/Const:output:0<dense_features/userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre1_indicator/SumSum4dense_features/userGenre1_indicator/one_hot:output:0Bdense_features/userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre1_indicator/ShapeShape0dense_features/userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre1_indicator/strided_sliceStridedSlice2dense_features/userGenre1_indicator/Shape:output:0@dense_features/userGenre1_indicator/strided_slice/stack:output:0Bdense_features/userGenre1_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre1_indicator/Reshape/shapePack:dense_features/userGenre1_indicator/strided_slice:output:0<dense_features/userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre1_indicator/ReshapeReshape0dense_features/userGenre1_indicator/Sum:output:0:dense_features/userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre2_indicator/ExpandDims
ExpandDimsinputs_usergenre2;dense_features/userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre2_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre2_indicator/ExpandDims:output:0Kdense_features/userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre2_indicator/to_sparse_input/indicesWhere@dense_features/userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre2_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre2_indicator/ExpandDims:output:0Cdense_features/userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre2_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre2_indicator/to_sparse_input/values:output:0Odense_features_usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre2_indicator/SparseToDenseSparseToDenseCdense_features/userGenre2_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre2_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre2_indicator/one_hotOneHot9dense_features/userGenre2_indicator/SparseToDense:dense:0:dense_features/userGenre2_indicator/one_hot/depth:output:0:dense_features/userGenre2_indicator/one_hot/Const:output:0<dense_features/userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre2_indicator/SumSum4dense_features/userGenre2_indicator/one_hot:output:0Bdense_features/userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre2_indicator/ShapeShape0dense_features/userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre2_indicator/strided_sliceStridedSlice2dense_features/userGenre2_indicator/Shape:output:0@dense_features/userGenre2_indicator/strided_slice/stack:output:0Bdense_features/userGenre2_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre2_indicator/Reshape/shapePack:dense_features/userGenre2_indicator/strided_slice:output:0<dense_features/userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre2_indicator/ReshapeReshape0dense_features/userGenre2_indicator/Sum:output:0:dense_features/userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre3_indicator/ExpandDims
ExpandDimsinputs_usergenre3;dense_features/userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre3_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre3_indicator/ExpandDims:output:0Kdense_features/userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre3_indicator/to_sparse_input/indicesWhere@dense_features/userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre3_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre3_indicator/ExpandDims:output:0Cdense_features/userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre3_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre3_indicator/to_sparse_input/values:output:0Odense_features_usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre3_indicator/SparseToDenseSparseToDenseCdense_features/userGenre3_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre3_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre3_indicator/one_hotOneHot9dense_features/userGenre3_indicator/SparseToDense:dense:0:dense_features/userGenre3_indicator/one_hot/depth:output:0:dense_features/userGenre3_indicator/one_hot/Const:output:0<dense_features/userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre3_indicator/SumSum4dense_features/userGenre3_indicator/one_hot:output:0Bdense_features/userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre3_indicator/ShapeShape0dense_features/userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre3_indicator/strided_sliceStridedSlice2dense_features/userGenre3_indicator/Shape:output:0@dense_features/userGenre3_indicator/strided_slice/stack:output:0Bdense_features/userGenre3_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre3_indicator/Reshape/shapePack:dense_features/userGenre3_indicator/strided_slice:output:0<dense_features/userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre3_indicator/ReshapeReshape0dense_features/userGenre3_indicator/Sum:output:0:dense_features/userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre4_indicator/ExpandDims
ExpandDimsinputs_usergenre4;dense_features/userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre4_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre4_indicator/ExpandDims:output:0Kdense_features/userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre4_indicator/to_sparse_input/indicesWhere@dense_features/userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre4_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre4_indicator/ExpandDims:output:0Cdense_features/userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre4_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre4_indicator/to_sparse_input/values:output:0Odense_features_usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre4_indicator/SparseToDenseSparseToDenseCdense_features/userGenre4_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre4_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre4_indicator/one_hotOneHot9dense_features/userGenre4_indicator/SparseToDense:dense:0:dense_features/userGenre4_indicator/one_hot/depth:output:0:dense_features/userGenre4_indicator/one_hot/Const:output:0<dense_features/userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre4_indicator/SumSum4dense_features/userGenre4_indicator/one_hot:output:0Bdense_features/userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre4_indicator/ShapeShape0dense_features/userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre4_indicator/strided_sliceStridedSlice2dense_features/userGenre4_indicator/Shape:output:0@dense_features/userGenre4_indicator/strided_slice/stack:output:0Bdense_features/userGenre4_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre4_indicator/Reshape/shapePack:dense_features/userGenre4_indicator/strided_slice:output:0<dense_features/userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre4_indicator/ReshapeReshape0dense_features/userGenre4_indicator/Sum:output:0:dense_features/userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2dense_features/userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.dense_features/userGenre5_indicator/ExpandDims
ExpandDimsinputs_usergenre5;dense_features/userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/userGenre5_indicator/to_sparse_input/NotEqualNotEqual7dense_features/userGenre5_indicator/ExpandDims:output:0Kdense_features/userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/userGenre5_indicator/to_sparse_input/indicesWhere@dense_features/userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/userGenre5_indicator/to_sparse_input/valuesGatherNd7dense_features/userGenre5_indicator/ExpandDims:output:0Cdense_features/userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/userGenre5_indicator/to_sparse_input/dense_shapeShape7dense_features/userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/userGenre5_indicator/to_sparse_input/values:output:0Odense_features_usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/userGenre5_indicator/SparseToDenseSparseToDenseCdense_features/userGenre5_indicator/to_sparse_input/indices:index:0Hdense_features/userGenre5_indicator/to_sparse_input/dense_shape:output:0Jdense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/userGenre5_indicator/one_hotOneHot9dense_features/userGenre5_indicator/SparseToDense:dense:0:dense_features/userGenre5_indicator/one_hot/depth:output:0:dense_features/userGenre5_indicator/one_hot/Const:output:0<dense_features/userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/userGenre5_indicator/SumSum4dense_features/userGenre5_indicator/one_hot:output:0Bdense_features/userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/userGenre5_indicator/ShapeShape0dense_features/userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/userGenre5_indicator/strided_sliceStridedSlice2dense_features/userGenre5_indicator/Shape:output:0@dense_features/userGenre5_indicator/strided_slice/stack:output:0Bdense_features/userGenre5_indicator/strided_slice/stack_1:output:0Bdense_features/userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/userGenre5_indicator/Reshape/shapePack:dense_features/userGenre5_indicator/strided_slice:output:0<dense_features/userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/userGenre5_indicator/ReshapeReshape0dense_features/userGenre5_indicator/Sum:output:0:dense_features/userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????x
-dense_features/userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)dense_features/userRatingCount/ExpandDims
ExpandDimsinputs_userratingcount6dense_features/userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
#dense_features/userRatingCount/CastCast2dense_features/userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????{
$dense_features/userRatingCount/ShapeShape'dense_features/userRatingCount/Cast:y:0*
T0*
_output_shapes
:|
2dense_features/userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,dense_features/userRatingCount/strided_sliceStridedSlice-dense_features/userRatingCount/Shape:output:0;dense_features/userRatingCount/strided_slice/stack:output:0=dense_features/userRatingCount/strided_slice/stack_1:output:0=dense_features/userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/userRatingCount/Reshape/shapePack5dense_features/userRatingCount/strided_slice:output:07dense_features/userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&dense_features/userRatingCount/ReshapeReshape'dense_features/userRatingCount/Cast:y:05dense_features/userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
.dense_features/userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*dense_features/userRatingStddev/ExpandDims
ExpandDimsinputs_userratingstddev7dense_features/userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
%dense_features/userRatingStddev/ShapeShape3dense_features/userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:}
3dense_features/userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/userRatingStddev/strided_sliceStridedSlice.dense_features/userRatingStddev/Shape:output:0<dense_features/userRatingStddev/strided_slice/stack:output:0>dense_features/userRatingStddev/strided_slice/stack_1:output:0>dense_features/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/userRatingStddev/Reshape/shapePack6dense_features/userRatingStddev/strided_slice:output:08dense_features/userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/userRatingStddev/ReshapeReshape3dense_features/userRatingStddev/ExpandDims:output:06dense_features/userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2.dense_features/movieAvgRating/Reshape:output:05dense_features/movieGenre1_indicator/Reshape:output:05dense_features/movieGenre2_indicator/Reshape:output:05dense_features/movieGenre3_indicator/Reshape:output:00dense_features/movieRatingCount/Reshape:output:01dense_features/movieRatingStddev/Reshape:output:0+dense_features/releaseYear/Reshape:output:0-dense_features/userAvgRating/Reshape:output:04dense_features/userGenre1_indicator/Reshape:output:04dense_features/userGenre2_indicator/Reshape:output:04dense_features/userGenre3_indicator/Reshape:output:04dense_features/userGenre4_indicator/Reshape:output:04dense_features/userGenre5_indicator/Reshape:output:0/dense_features/userRatingCount/Reshape:output:00dense_features/userRatingStddev/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
fm_layer/MatMul/ReadVariableOpReadVariableOp'fm_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
fm_layer/MatMulMatMuldense_features/concat:output:0&fm_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
S
fm_layer/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
fm_layer/PowPowfm_layer/MatMul:product:0fm_layer/Pow/y:output:0*
T0*'
_output_shapes
:?????????
U
fm_layer/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
fm_layer/Pow_1Powdense_features/concat:output:0fm_layer/Pow_1/y:output:0*
T0*(
_output_shapes
:???????????
fm_layer/Pow_2/ReadVariableOpReadVariableOp'fm_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0U
fm_layer/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
fm_layer/Pow_2Pow%fm_layer/Pow_2/ReadVariableOp:value:0fm_layer/Pow_2/y:output:0*
T0*
_output_shapes
:	?
u
fm_layer/MatMul_1MatMulfm_layer/Pow_1:z:0fm_layer/Pow_2:z:0*
T0*'
_output_shapes
:?????????
t
fm_layer/subSubfm_layer/Pow:z:0fm_layer/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
`
fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
fm_layer/SumSumfm_layer/sub:z:0'fm_layer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(S
fm_layer/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
fm_layer/mulMulfm_layer/Sum:output:0fm_layer/mul/y:output:0*
T0*'
_output_shapes
:?????????l
add/addAddV2dense/BiasAdd:output:0fm_layer/mul:z:0*
T0*'
_output_shapes
:?????????\
activation/SigmoidSigmoidadd/add:z:0*
T0*'
_output_shapes
:??????????
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOpC^dense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2C^dense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2C^dense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2B^dense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2^fm_layer/MatMul/ReadVariableOp^fm_layer/Pow_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2?
Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV2Bdense_features/movieGenre1_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV2Bdense_features/movieGenre2_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV2Bdense_features/movieGenre3_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre1_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre2_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre3_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre4_indicator/None_Lookup/LookupTableFindV22?
Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV2Adense_features/userGenre5_indicator/None_Lookup/LookupTableFindV22@
fm_layer/MatMul/ReadVariableOpfm_layer/MatMul/ReadVariableOp2>
fm_layer/Pow_2/ReadVariableOpfm_layer/Pow_2/ReadVariableOp:Z V
#
_output_shapes
:?????????
/
_user_specified_nameinputs/movieAvgRating:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre1:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre2:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre3:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/movieRatingCount:]Y
#
_output_shapes
:?????????
2
_user_specified_nameinputs/movieRatingStddev:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/releaseYear:YU
#
_output_shapes
:?????????
.
_user_specified_nameinputs/userAvgRating:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre1:V	R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre2:V
R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre3:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre4:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre5:[W
#
_output_shapes
:?????????
0
_user_specified_nameinputs/userRatingCount:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_12091
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
O
#__inference_add_layer_call_fn_12035
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_9502`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
j
>__inference_add_layer_call_and_return_conditional_losses_12041
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
,
__inference__destroyer_12109
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
,
__inference__destroyer_12181
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_121222
.table_init407_lookuptableimportv2_table_handle*
&table_init407_lookuptableimportv2_keys,
(table_init407_lookuptableimportv2_values	
identity??!table_init407/LookupTableImportV2?
!table_init407/LookupTableImportV2LookupTableImportV2.table_init407_lookuptableimportv2_table_handle&table_init407_lookuptableimportv2_keys(table_init407_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init407/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init407/LookupTableImportV2!table_init407/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
:
__inference__creator_12132
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name486*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
:
__inference__creator_12096
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name372*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
H__inference_dense_features_layer_call_and_return_conditional_losses_9410
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14D
@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	
identity??3movieGenre1_indicator/None_Lookup/LookupTableFindV2?3movieGenre2_indicator/None_Lookup/LookupTableFindV2?3movieGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre1_indicator/None_Lookup/LookupTableFindV2?2userGenre2_indicator/None_Lookup/LookupTableFindV2?2userGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre4_indicator/None_Lookup/LookupTableFindV2?2userGenre5_indicator/None_Lookup/LookupTableFindV2h
movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieAvgRating/ExpandDims
ExpandDimsfeatures&movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????f
movieAvgRating/ShapeShape"movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:l
"movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieAvgRating/strided_sliceStridedSlicemovieAvgRating/Shape:output:0+movieAvgRating/strided_slice/stack:output:0-movieAvgRating/strided_slice/stack_1:output:0-movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieAvgRating/Reshape/shapePack%movieAvgRating/strided_slice:output:0'movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieAvgRating/ReshapeReshape"movieAvgRating/ExpandDims:output:0%movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre1_indicator/ExpandDims
ExpandDims
features_1-movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre1_indicator/to_sparse_input/NotEqualNotEqual)movieGenre1_indicator/ExpandDims:output:0=movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre1_indicator/to_sparse_input/indicesWhere2movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre1_indicator/to_sparse_input/valuesGatherNd)movieGenre1_indicator/ExpandDims:output:05movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre1_indicator/to_sparse_input/dense_shapeShape)movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre1_indicator/to_sparse_input/values:output:0Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre1_indicator/SparseToDenseSparseToDense5movieGenre1_indicator/to_sparse_input/indices:index:0:movieGenre1_indicator/to_sparse_input/dense_shape:output:0<movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre1_indicator/one_hotOneHot+movieGenre1_indicator/SparseToDense:dense:0,movieGenre1_indicator/one_hot/depth:output:0,movieGenre1_indicator/one_hot/Const:output:0.movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre1_indicator/SumSum&movieGenre1_indicator/one_hot:output:04movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre1_indicator/ShapeShape"movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre1_indicator/strided_sliceStridedSlice$movieGenre1_indicator/Shape:output:02movieGenre1_indicator/strided_slice/stack:output:04movieGenre1_indicator/strided_slice/stack_1:output:04movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre1_indicator/Reshape/shapePack,movieGenre1_indicator/strided_slice:output:0.movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre1_indicator/ReshapeReshape"movieGenre1_indicator/Sum:output:0,movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre2_indicator/ExpandDims
ExpandDims
features_2-movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre2_indicator/to_sparse_input/NotEqualNotEqual)movieGenre2_indicator/ExpandDims:output:0=movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre2_indicator/to_sparse_input/indicesWhere2movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre2_indicator/to_sparse_input/valuesGatherNd)movieGenre2_indicator/ExpandDims:output:05movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre2_indicator/to_sparse_input/dense_shapeShape)movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre2_indicator/to_sparse_input/values:output:0Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre2_indicator/SparseToDenseSparseToDense5movieGenre2_indicator/to_sparse_input/indices:index:0:movieGenre2_indicator/to_sparse_input/dense_shape:output:0<movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre2_indicator/one_hotOneHot+movieGenre2_indicator/SparseToDense:dense:0,movieGenre2_indicator/one_hot/depth:output:0,movieGenre2_indicator/one_hot/Const:output:0.movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre2_indicator/SumSum&movieGenre2_indicator/one_hot:output:04movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre2_indicator/ShapeShape"movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre2_indicator/strided_sliceStridedSlice$movieGenre2_indicator/Shape:output:02movieGenre2_indicator/strided_slice/stack:output:04movieGenre2_indicator/strided_slice/stack_1:output:04movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre2_indicator/Reshape/shapePack,movieGenre2_indicator/strided_slice:output:0.movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre2_indicator/ReshapeReshape"movieGenre2_indicator/Sum:output:0,movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre3_indicator/ExpandDims
ExpandDims
features_3-movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre3_indicator/to_sparse_input/NotEqualNotEqual)movieGenre3_indicator/ExpandDims:output:0=movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre3_indicator/to_sparse_input/indicesWhere2movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre3_indicator/to_sparse_input/valuesGatherNd)movieGenre3_indicator/ExpandDims:output:05movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre3_indicator/to_sparse_input/dense_shapeShape)movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre3_indicator/to_sparse_input/values:output:0Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre3_indicator/SparseToDenseSparseToDense5movieGenre3_indicator/to_sparse_input/indices:index:0:movieGenre3_indicator/to_sparse_input/dense_shape:output:0<movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre3_indicator/one_hotOneHot+movieGenre3_indicator/SparseToDense:dense:0,movieGenre3_indicator/one_hot/depth:output:0,movieGenre3_indicator/one_hot/Const:output:0.movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre3_indicator/SumSum&movieGenre3_indicator/one_hot:output:04movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre3_indicator/ShapeShape"movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre3_indicator/strided_sliceStridedSlice$movieGenre3_indicator/Shape:output:02movieGenre3_indicator/strided_slice/stack:output:04movieGenre3_indicator/strided_slice/stack_1:output:04movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre3_indicator/Reshape/shapePack,movieGenre3_indicator/strided_slice:output:0.movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre3_indicator/ReshapeReshape"movieGenre3_indicator/Sum:output:0,movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingCount/ExpandDims
ExpandDims
features_4(movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
movieRatingCount/CastCast$movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????_
movieRatingCount/ShapeShapemovieRatingCount/Cast:y:0*
T0*
_output_shapes
:n
$movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingCount/strided_sliceStridedSlicemovieRatingCount/Shape:output:0-movieRatingCount/strided_slice/stack:output:0/movieRatingCount/strided_slice/stack_1:output:0/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingCount/Reshape/shapePack'movieRatingCount/strided_slice:output:0)movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingCount/ReshapeReshapemovieRatingCount/Cast:y:0'movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingStddev/ExpandDims
ExpandDims
features_5)movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????l
movieRatingStddev/ShapeShape%movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:o
%movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingStddev/strided_sliceStridedSlice movieRatingStddev/Shape:output:0.movieRatingStddev/strided_slice/stack:output:00movieRatingStddev/strided_slice/stack_1:output:00movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingStddev/Reshape/shapePack(movieRatingStddev/strided_slice:output:0*movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingStddev/ReshapeReshape%movieRatingStddev/ExpandDims:output:0(movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
releaseYear/ExpandDims
ExpandDims
features_6#releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????z
releaseYear/CastCastreleaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????U
releaseYear/ShapeShapereleaseYear/Cast:y:0*
T0*
_output_shapes
:i
releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
releaseYear/strided_sliceStridedSlicereleaseYear/Shape:output:0(releaseYear/strided_slice/stack:output:0*releaseYear/strided_slice/stack_1:output:0*releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
releaseYear/Reshape/shapePack"releaseYear/strided_slice:output:0$releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
releaseYear/ReshapeReshapereleaseYear/Cast:y:0"releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????g
userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userAvgRating/ExpandDims
ExpandDims
features_7%userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????d
userAvgRating/ShapeShape!userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:k
!userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userAvgRating/strided_sliceStridedSliceuserAvgRating/Shape:output:0*userAvgRating/strided_slice/stack:output:0,userAvgRating/strided_slice/stack_1:output:0,userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userAvgRating/Reshape/shapePack$userAvgRating/strided_slice:output:0&userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userAvgRating/ReshapeReshape!userAvgRating/ExpandDims:output:0$userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre1_indicator/ExpandDims
ExpandDims
features_8,userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre1_indicator/to_sparse_input/NotEqualNotEqual(userGenre1_indicator/ExpandDims:output:0<userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre1_indicator/to_sparse_input/indicesWhere1userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre1_indicator/to_sparse_input/valuesGatherNd(userGenre1_indicator/ExpandDims:output:04userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre1_indicator/to_sparse_input/dense_shapeShape(userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre1_indicator/to_sparse_input/values:output:0@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre1_indicator/SparseToDenseSparseToDense4userGenre1_indicator/to_sparse_input/indices:index:09userGenre1_indicator/to_sparse_input/dense_shape:output:0;userGenre1_indicator/None_Lookup/LookupTableFindV2:values:09userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre1_indicator/one_hotOneHot*userGenre1_indicator/SparseToDense:dense:0+userGenre1_indicator/one_hot/depth:output:0+userGenre1_indicator/one_hot/Const:output:0-userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre1_indicator/SumSum%userGenre1_indicator/one_hot:output:03userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre1_indicator/ShapeShape!userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre1_indicator/strided_sliceStridedSlice#userGenre1_indicator/Shape:output:01userGenre1_indicator/strided_slice/stack:output:03userGenre1_indicator/strided_slice/stack_1:output:03userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre1_indicator/Reshape/shapePack+userGenre1_indicator/strided_slice:output:0-userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre1_indicator/ReshapeReshape!userGenre1_indicator/Sum:output:0+userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre2_indicator/ExpandDims
ExpandDims
features_9,userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre2_indicator/to_sparse_input/NotEqualNotEqual(userGenre2_indicator/ExpandDims:output:0<userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre2_indicator/to_sparse_input/indicesWhere1userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre2_indicator/to_sparse_input/valuesGatherNd(userGenre2_indicator/ExpandDims:output:04userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre2_indicator/to_sparse_input/dense_shapeShape(userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre2_indicator/to_sparse_input/values:output:0@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre2_indicator/SparseToDenseSparseToDense4userGenre2_indicator/to_sparse_input/indices:index:09userGenre2_indicator/to_sparse_input/dense_shape:output:0;userGenre2_indicator/None_Lookup/LookupTableFindV2:values:09userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre2_indicator/one_hotOneHot*userGenre2_indicator/SparseToDense:dense:0+userGenre2_indicator/one_hot/depth:output:0+userGenre2_indicator/one_hot/Const:output:0-userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre2_indicator/SumSum%userGenre2_indicator/one_hot:output:03userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre2_indicator/ShapeShape!userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre2_indicator/strided_sliceStridedSlice#userGenre2_indicator/Shape:output:01userGenre2_indicator/strided_slice/stack:output:03userGenre2_indicator/strided_slice/stack_1:output:03userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre2_indicator/Reshape/shapePack+userGenre2_indicator/strided_slice:output:0-userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre2_indicator/ReshapeReshape!userGenre2_indicator/Sum:output:0+userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre3_indicator/ExpandDims
ExpandDimsfeatures_10,userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre3_indicator/to_sparse_input/NotEqualNotEqual(userGenre3_indicator/ExpandDims:output:0<userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre3_indicator/to_sparse_input/indicesWhere1userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre3_indicator/to_sparse_input/valuesGatherNd(userGenre3_indicator/ExpandDims:output:04userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre3_indicator/to_sparse_input/dense_shapeShape(userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre3_indicator/to_sparse_input/values:output:0@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre3_indicator/SparseToDenseSparseToDense4userGenre3_indicator/to_sparse_input/indices:index:09userGenre3_indicator/to_sparse_input/dense_shape:output:0;userGenre3_indicator/None_Lookup/LookupTableFindV2:values:09userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre3_indicator/one_hotOneHot*userGenre3_indicator/SparseToDense:dense:0+userGenre3_indicator/one_hot/depth:output:0+userGenre3_indicator/one_hot/Const:output:0-userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre3_indicator/SumSum%userGenre3_indicator/one_hot:output:03userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre3_indicator/ShapeShape!userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre3_indicator/strided_sliceStridedSlice#userGenre3_indicator/Shape:output:01userGenre3_indicator/strided_slice/stack:output:03userGenre3_indicator/strided_slice/stack_1:output:03userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre3_indicator/Reshape/shapePack+userGenre3_indicator/strided_slice:output:0-userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre3_indicator/ReshapeReshape!userGenre3_indicator/Sum:output:0+userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre4_indicator/ExpandDims
ExpandDimsfeatures_11,userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre4_indicator/to_sparse_input/NotEqualNotEqual(userGenre4_indicator/ExpandDims:output:0<userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre4_indicator/to_sparse_input/indicesWhere1userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre4_indicator/to_sparse_input/valuesGatherNd(userGenre4_indicator/ExpandDims:output:04userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre4_indicator/to_sparse_input/dense_shapeShape(userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre4_indicator/to_sparse_input/values:output:0@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre4_indicator/SparseToDenseSparseToDense4userGenre4_indicator/to_sparse_input/indices:index:09userGenre4_indicator/to_sparse_input/dense_shape:output:0;userGenre4_indicator/None_Lookup/LookupTableFindV2:values:09userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre4_indicator/one_hotOneHot*userGenre4_indicator/SparseToDense:dense:0+userGenre4_indicator/one_hot/depth:output:0+userGenre4_indicator/one_hot/Const:output:0-userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre4_indicator/SumSum%userGenre4_indicator/one_hot:output:03userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre4_indicator/ShapeShape!userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre4_indicator/strided_sliceStridedSlice#userGenre4_indicator/Shape:output:01userGenre4_indicator/strided_slice/stack:output:03userGenre4_indicator/strided_slice/stack_1:output:03userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre4_indicator/Reshape/shapePack+userGenre4_indicator/strided_slice:output:0-userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre4_indicator/ReshapeReshape!userGenre4_indicator/Sum:output:0+userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre5_indicator/ExpandDims
ExpandDimsfeatures_12,userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre5_indicator/to_sparse_input/NotEqualNotEqual(userGenre5_indicator/ExpandDims:output:0<userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre5_indicator/to_sparse_input/indicesWhere1userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre5_indicator/to_sparse_input/valuesGatherNd(userGenre5_indicator/ExpandDims:output:04userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre5_indicator/to_sparse_input/dense_shapeShape(userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre5_indicator/to_sparse_input/values:output:0@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre5_indicator/SparseToDenseSparseToDense4userGenre5_indicator/to_sparse_input/indices:index:09userGenre5_indicator/to_sparse_input/dense_shape:output:0;userGenre5_indicator/None_Lookup/LookupTableFindV2:values:09userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre5_indicator/one_hotOneHot*userGenre5_indicator/SparseToDense:dense:0+userGenre5_indicator/one_hot/depth:output:0+userGenre5_indicator/one_hot/Const:output:0-userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre5_indicator/SumSum%userGenre5_indicator/one_hot:output:03userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre5_indicator/ShapeShape!userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre5_indicator/strided_sliceStridedSlice#userGenre5_indicator/Shape:output:01userGenre5_indicator/strided_slice/stack:output:03userGenre5_indicator/strided_slice/stack_1:output:03userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre5_indicator/Reshape/shapePack+userGenre5_indicator/strided_slice:output:0-userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre5_indicator/ReshapeReshape!userGenre5_indicator/Sum:output:0+userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingCount/ExpandDims
ExpandDimsfeatures_13'userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
userRatingCount/CastCast#userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????]
userRatingCount/ShapeShapeuserRatingCount/Cast:y:0*
T0*
_output_shapes
:m
#userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingCount/strided_sliceStridedSliceuserRatingCount/Shape:output:0,userRatingCount/strided_slice/stack:output:0.userRatingCount/strided_slice/stack_1:output:0.userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingCount/Reshape/shapePack&userRatingCount/strided_slice:output:0(userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingCount/ReshapeReshapeuserRatingCount/Cast:y:0&userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingStddev/ExpandDims
ExpandDimsfeatures_14(userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ShapeShape$userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:n
$userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingStddev/strided_sliceStridedSliceuserRatingStddev/Shape:output:0-userRatingStddev/strided_slice/stack:output:0/userRatingStddev/strided_slice/stack_1:output:0/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingStddev/Reshape/shapePack'userRatingStddev/strided_slice:output:0)userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingStddev/ReshapeReshape$userRatingStddev/ExpandDims:output:0'userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2movieAvgRating/Reshape:output:0&movieGenre1_indicator/Reshape:output:0&movieGenre2_indicator/Reshape:output:0&movieGenre3_indicator/Reshape:output:0!movieRatingCount/Reshape:output:0"movieRatingStddev/Reshape:output:0releaseYear/Reshape:output:0userAvgRating/Reshape:output:0%userGenre1_indicator/Reshape:output:0%userGenre2_indicator/Reshape:output:0%userGenre3_indicator/Reshape:output:0%userGenre4_indicator/Reshape:output:0%userGenre5_indicator/Reshape:output:0 userRatingCount/Reshape:output:0!userRatingStddev/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp4^movieGenre1_indicator/None_Lookup/LookupTableFindV24^movieGenre2_indicator/None_Lookup/LookupTableFindV24^movieGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre1_indicator/None_Lookup/LookupTableFindV23^userGenre2_indicator/None_Lookup/LookupTableFindV23^userGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre4_indicator/None_Lookup/LookupTableFindV23^userGenre5_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : 2j
3movieGenre1_indicator/None_Lookup/LookupTableFindV23movieGenre1_indicator/None_Lookup/LookupTableFindV22j
3movieGenre2_indicator/None_Lookup/LookupTableFindV23movieGenre2_indicator/None_Lookup/LookupTableFindV22j
3movieGenre3_indicator/None_Lookup/LookupTableFindV23movieGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre1_indicator/None_Lookup/LookupTableFindV22userGenre1_indicator/None_Lookup/LookupTableFindV22h
2userGenre2_indicator/None_Lookup/LookupTableFindV22userGenre2_indicator/None_Lookup/LookupTableFindV22h
2userGenre3_indicator/None_Lookup/LookupTableFindV22userGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre4_indicator/None_Lookup/LookupTableFindV22userGenre4_indicator/None_Lookup/LookupTableFindV22h
2userGenre5_indicator/None_Lookup/LookupTableFindV22userGenre5_indicator/None_Lookup/LookupTableFindV2:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:M	I
#
_output_shapes
:?????????
"
_user_specified_name
features:M
I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
*__inference_activation_layer_call_fn_12046

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_9509`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_10593
inputs_movieavgrating
inputs_moviegenre1
inputs_moviegenre2
inputs_moviegenre3
inputs_movieratingcount
inputs_movieratingstddev
inputs_releaseyear
inputs_useravgrating
inputs_usergenre1
inputs_usergenre2
inputs_usergenre3
inputs_usergenre4
inputs_usergenre5
inputs_userratingcount
inputs_userratingstddev
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15:	?

unknown_16:

unknown_17:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_movieavgratinginputs_moviegenre1inputs_moviegenre2inputs_moviegenre3inputs_movieratingcountinputs_movieratingstddevinputs_releaseyearinputs_useravgratinginputs_usergenre1inputs_usergenre2inputs_usergenre3inputs_usergenre4inputs_usergenre5inputs_userratingcountinputs_userratingstddevunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*-
Tin&
$2"								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
 !*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_nameinputs/movieAvgRating:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre1:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre2:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/movieGenre3:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/movieRatingCount:]Y
#
_output_shapes
:?????????
2
_user_specified_nameinputs/movieRatingStddev:WS
#
_output_shapes
:?????????
,
_user_specified_nameinputs/releaseYear:YU
#
_output_shapes
:?????????
.
_user_specified_nameinputs/userAvgRating:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre1:V	R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre2:V
R
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre3:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre4:VR
#
_output_shapes
:?????????
+
_user_specified_nameinputs/userGenre5:[W
#
_output_shapes
:?????????
0
_user_specified_nameinputs/userRatingCount:\X
#
_output_shapes
:?????????
1
_user_specified_nameinputs/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_122572
.table_init521_lookuptableimportv2_table_handle*
&table_init521_lookuptableimportv2_keys,
(table_init521_lookuptableimportv2_values	
identity??!table_init521/LookupTableImportV2?
!table_init521/LookupTableImportV2LookupTableImportV2.table_init521_lookuptableimportv2_table_handle&table_init521_lookuptableimportv2_keys(table_init521_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init521/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init521/LookupTableImportV2!table_init521/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
??
?
I__inference_dense_features_layer_call_and_return_conditional_losses_11971
features_movieavgrating
features_moviegenre1
features_moviegenre2
features_moviegenre3
features_movieratingcount
features_movieratingstddev
features_releaseyear
features_useravgrating
features_usergenre1
features_usergenre2
features_usergenre3
features_usergenre4
features_usergenre5
features_userratingcount
features_userratingstddevD
@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value	D
@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handleE
Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value	C
?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handleD
@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value	
identity??3movieGenre1_indicator/None_Lookup/LookupTableFindV2?3movieGenre2_indicator/None_Lookup/LookupTableFindV2?3movieGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre1_indicator/None_Lookup/LookupTableFindV2?2userGenre2_indicator/None_Lookup/LookupTableFindV2?2userGenre3_indicator/None_Lookup/LookupTableFindV2?2userGenre4_indicator/None_Lookup/LookupTableFindV2?2userGenre5_indicator/None_Lookup/LookupTableFindV2h
movieAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieAvgRating/ExpandDims
ExpandDimsfeatures_movieavgrating&movieAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????f
movieAvgRating/ShapeShape"movieAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:l
"movieAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$movieAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$movieAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieAvgRating/strided_sliceStridedSlicemovieAvgRating/Shape:output:0+movieAvgRating/strided_slice/stack:output:0-movieAvgRating/strided_slice/stack_1:output:0-movieAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
movieAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieAvgRating/Reshape/shapePack%movieAvgRating/strided_slice:output:0'movieAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieAvgRating/ReshapeReshape"movieAvgRating/ExpandDims:output:0%movieAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre1_indicator/ExpandDims
ExpandDimsfeatures_moviegenre1-movieGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre1_indicator/to_sparse_input/NotEqualNotEqual)movieGenre1_indicator/ExpandDims:output:0=movieGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre1_indicator/to_sparse_input/indicesWhere2movieGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre1_indicator/to_sparse_input/valuesGatherNd)movieGenre1_indicator/ExpandDims:output:05movieGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre1_indicator/to_sparse_input/dense_shapeShape)movieGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre1_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre1_indicator/to_sparse_input/values:output:0Amoviegenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre1_indicator/SparseToDenseSparseToDense5movieGenre1_indicator/to_sparse_input/indices:index:0:movieGenre1_indicator/to_sparse_input/dense_shape:output:0<movieGenre1_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre1_indicator/one_hotOneHot+movieGenre1_indicator/SparseToDense:dense:0,movieGenre1_indicator/one_hot/depth:output:0,movieGenre1_indicator/one_hot/Const:output:0.movieGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre1_indicator/SumSum&movieGenre1_indicator/one_hot:output:04movieGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre1_indicator/ShapeShape"movieGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre1_indicator/strided_sliceStridedSlice$movieGenre1_indicator/Shape:output:02movieGenre1_indicator/strided_slice/stack:output:04movieGenre1_indicator/strided_slice/stack_1:output:04movieGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre1_indicator/Reshape/shapePack,movieGenre1_indicator/strided_slice:output:0.movieGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre1_indicator/ReshapeReshape"movieGenre1_indicator/Sum:output:0,movieGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre2_indicator/ExpandDims
ExpandDimsfeatures_moviegenre2-movieGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre2_indicator/to_sparse_input/NotEqualNotEqual)movieGenre2_indicator/ExpandDims:output:0=movieGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre2_indicator/to_sparse_input/indicesWhere2movieGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre2_indicator/to_sparse_input/valuesGatherNd)movieGenre2_indicator/ExpandDims:output:05movieGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre2_indicator/to_sparse_input/dense_shapeShape)movieGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre2_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre2_indicator/to_sparse_input/values:output:0Amoviegenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre2_indicator/SparseToDenseSparseToDense5movieGenre2_indicator/to_sparse_input/indices:index:0:movieGenre2_indicator/to_sparse_input/dense_shape:output:0<movieGenre2_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre2_indicator/one_hotOneHot+movieGenre2_indicator/SparseToDense:dense:0,movieGenre2_indicator/one_hot/depth:output:0,movieGenre2_indicator/one_hot/Const:output:0.movieGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre2_indicator/SumSum&movieGenre2_indicator/one_hot:output:04movieGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre2_indicator/ShapeShape"movieGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre2_indicator/strided_sliceStridedSlice$movieGenre2_indicator/Shape:output:02movieGenre2_indicator/strided_slice/stack:output:04movieGenre2_indicator/strided_slice/stack_1:output:04movieGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre2_indicator/Reshape/shapePack,movieGenre2_indicator/strided_slice:output:0.movieGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre2_indicator/ReshapeReshape"movieGenre2_indicator/Sum:output:0,movieGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o
$movieGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 movieGenre3_indicator/ExpandDims
ExpandDimsfeatures_moviegenre3-movieGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????u
4movieGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.movieGenre3_indicator/to_sparse_input/NotEqualNotEqual)movieGenre3_indicator/ExpandDims:output:0=movieGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-movieGenre3_indicator/to_sparse_input/indicesWhere2movieGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,movieGenre3_indicator/to_sparse_input/valuesGatherNd)movieGenre3_indicator/ExpandDims:output:05movieGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1movieGenre3_indicator/to_sparse_input/dense_shapeShape)movieGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
3movieGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@moviegenre3_indicator_none_lookup_lookuptablefindv2_table_handle5movieGenre3_indicator/to_sparse_input/values:output:0Amoviegenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1movieGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#movieGenre3_indicator/SparseToDenseSparseToDense5movieGenre3_indicator/to_sparse_input/indices:index:0:movieGenre3_indicator/to_sparse_input/dense_shape:output:0<movieGenre3_indicator/None_Lookup/LookupTableFindV2:values:0:movieGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#movieGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%movieGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#movieGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
movieGenre3_indicator/one_hotOneHot+movieGenre3_indicator/SparseToDense:dense:0,movieGenre3_indicator/one_hot/depth:output:0,movieGenre3_indicator/one_hot/Const:output:0.movieGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+movieGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
movieGenre3_indicator/SumSum&movieGenre3_indicator/one_hot:output:04movieGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
movieGenre3_indicator/ShapeShape"movieGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:s
)movieGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+movieGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+movieGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#movieGenre3_indicator/strided_sliceStridedSlice$movieGenre3_indicator/Shape:output:02movieGenre3_indicator/strided_slice/stack:output:04movieGenre3_indicator/strided_slice/stack_1:output:04movieGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%movieGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#movieGenre3_indicator/Reshape/shapePack,movieGenre3_indicator/strided_slice:output:0.movieGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieGenre3_indicator/ReshapeReshape"movieGenre3_indicator/Sum:output:0,movieGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
movieRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingCount/ExpandDims
ExpandDimsfeatures_movieratingcount(movieRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
movieRatingCount/CastCast$movieRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????_
movieRatingCount/ShapeShapemovieRatingCount/Cast:y:0*
T0*
_output_shapes
:n
$movieRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&movieRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&movieRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingCount/strided_sliceStridedSlicemovieRatingCount/Shape:output:0-movieRatingCount/strided_slice/stack:output:0/movieRatingCount/strided_slice/stack_1:output:0/movieRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 movieRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingCount/Reshape/shapePack'movieRatingCount/strided_slice:output:0)movieRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingCount/ReshapeReshapemovieRatingCount/Cast:y:0'movieRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 movieRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
movieRatingStddev/ExpandDims
ExpandDimsfeatures_movieratingstddev)movieRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????l
movieRatingStddev/ShapeShape%movieRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:o
%movieRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
movieRatingStddev/strided_sliceStridedSlice movieRatingStddev/Shape:output:0.movieRatingStddev/strided_slice/stack:output:00movieRatingStddev/strided_slice/stack_1:output:00movieRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
movieRatingStddev/Reshape/shapePack(movieRatingStddev/strided_slice:output:0*movieRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
movieRatingStddev/ReshapeReshape%movieRatingStddev/ExpandDims:output:0(movieRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
releaseYear/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
releaseYear/ExpandDims
ExpandDimsfeatures_releaseyear#releaseYear/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????z
releaseYear/CastCastreleaseYear/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????U
releaseYear/ShapeShapereleaseYear/Cast:y:0*
T0*
_output_shapes
:i
releaseYear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!releaseYear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!releaseYear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
releaseYear/strided_sliceStridedSlicereleaseYear/Shape:output:0(releaseYear/strided_slice/stack:output:0*releaseYear/strided_slice/stack_1:output:0*releaseYear/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
releaseYear/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
releaseYear/Reshape/shapePack"releaseYear/strided_slice:output:0$releaseYear/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
releaseYear/ReshapeReshapereleaseYear/Cast:y:0"releaseYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????g
userAvgRating/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userAvgRating/ExpandDims
ExpandDimsfeatures_useravgrating%userAvgRating/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????d
userAvgRating/ShapeShape!userAvgRating/ExpandDims:output:0*
T0*
_output_shapes
:k
!userAvgRating/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#userAvgRating/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#userAvgRating/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userAvgRating/strided_sliceStridedSliceuserAvgRating/Shape:output:0*userAvgRating/strided_slice/stack:output:0,userAvgRating/strided_slice/stack_1:output:0,userAvgRating/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
userAvgRating/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userAvgRating/Reshape/shapePack$userAvgRating/strided_slice:output:0&userAvgRating/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userAvgRating/ReshapeReshape!userAvgRating/ExpandDims:output:0$userAvgRating/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre1_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre1_indicator/ExpandDims
ExpandDimsfeatures_usergenre1,userGenre1_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre1_indicator/to_sparse_input/NotEqualNotEqual(userGenre1_indicator/ExpandDims:output:0<userGenre1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre1_indicator/to_sparse_input/indicesWhere1userGenre1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre1_indicator/to_sparse_input/valuesGatherNd(userGenre1_indicator/ExpandDims:output:04userGenre1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre1_indicator/to_sparse_input/dense_shapeShape(userGenre1_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre1_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre1_indicator/to_sparse_input/values:output:0@usergenre1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre1_indicator/SparseToDenseSparseToDense4userGenre1_indicator/to_sparse_input/indices:index:09userGenre1_indicator/to_sparse_input/dense_shape:output:0;userGenre1_indicator/None_Lookup/LookupTableFindV2:values:09userGenre1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre1_indicator/one_hotOneHot*userGenre1_indicator/SparseToDense:dense:0+userGenre1_indicator/one_hot/depth:output:0+userGenre1_indicator/one_hot/Const:output:0-userGenre1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre1_indicator/SumSum%userGenre1_indicator/one_hot:output:03userGenre1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre1_indicator/ShapeShape!userGenre1_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre1_indicator/strided_sliceStridedSlice#userGenre1_indicator/Shape:output:01userGenre1_indicator/strided_slice/stack:output:03userGenre1_indicator/strided_slice/stack_1:output:03userGenre1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre1_indicator/Reshape/shapePack+userGenre1_indicator/strided_slice:output:0-userGenre1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre1_indicator/ReshapeReshape!userGenre1_indicator/Sum:output:0+userGenre1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre2_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre2_indicator/ExpandDims
ExpandDimsfeatures_usergenre2,userGenre2_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre2_indicator/to_sparse_input/NotEqualNotEqual(userGenre2_indicator/ExpandDims:output:0<userGenre2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre2_indicator/to_sparse_input/indicesWhere1userGenre2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre2_indicator/to_sparse_input/valuesGatherNd(userGenre2_indicator/ExpandDims:output:04userGenre2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre2_indicator/to_sparse_input/dense_shapeShape(userGenre2_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre2_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre2_indicator/to_sparse_input/values:output:0@usergenre2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre2_indicator/SparseToDenseSparseToDense4userGenre2_indicator/to_sparse_input/indices:index:09userGenre2_indicator/to_sparse_input/dense_shape:output:0;userGenre2_indicator/None_Lookup/LookupTableFindV2:values:09userGenre2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre2_indicator/one_hotOneHot*userGenre2_indicator/SparseToDense:dense:0+userGenre2_indicator/one_hot/depth:output:0+userGenre2_indicator/one_hot/Const:output:0-userGenre2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre2_indicator/SumSum%userGenre2_indicator/one_hot:output:03userGenre2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre2_indicator/ShapeShape!userGenre2_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre2_indicator/strided_sliceStridedSlice#userGenre2_indicator/Shape:output:01userGenre2_indicator/strided_slice/stack:output:03userGenre2_indicator/strided_slice/stack_1:output:03userGenre2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre2_indicator/Reshape/shapePack+userGenre2_indicator/strided_slice:output:0-userGenre2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre2_indicator/ReshapeReshape!userGenre2_indicator/Sum:output:0+userGenre2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre3_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre3_indicator/ExpandDims
ExpandDimsfeatures_usergenre3,userGenre3_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre3_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre3_indicator/to_sparse_input/NotEqualNotEqual(userGenre3_indicator/ExpandDims:output:0<userGenre3_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre3_indicator/to_sparse_input/indicesWhere1userGenre3_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre3_indicator/to_sparse_input/valuesGatherNd(userGenre3_indicator/ExpandDims:output:04userGenre3_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre3_indicator/to_sparse_input/dense_shapeShape(userGenre3_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre3_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre3_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre3_indicator/to_sparse_input/values:output:0@usergenre3_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre3_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre3_indicator/SparseToDenseSparseToDense4userGenre3_indicator/to_sparse_input/indices:index:09userGenre3_indicator/to_sparse_input/dense_shape:output:0;userGenre3_indicator/None_Lookup/LookupTableFindV2:values:09userGenre3_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre3_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre3_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre3_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre3_indicator/one_hotOneHot*userGenre3_indicator/SparseToDense:dense:0+userGenre3_indicator/one_hot/depth:output:0+userGenre3_indicator/one_hot/Const:output:0-userGenre3_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre3_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre3_indicator/SumSum%userGenre3_indicator/one_hot:output:03userGenre3_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre3_indicator/ShapeShape!userGenre3_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre3_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre3_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre3_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre3_indicator/strided_sliceStridedSlice#userGenre3_indicator/Shape:output:01userGenre3_indicator/strided_slice/stack:output:03userGenre3_indicator/strided_slice/stack_1:output:03userGenre3_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre3_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre3_indicator/Reshape/shapePack+userGenre3_indicator/strided_slice:output:0-userGenre3_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre3_indicator/ReshapeReshape!userGenre3_indicator/Sum:output:0+userGenre3_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre4_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre4_indicator/ExpandDims
ExpandDimsfeatures_usergenre4,userGenre4_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre4_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre4_indicator/to_sparse_input/NotEqualNotEqual(userGenre4_indicator/ExpandDims:output:0<userGenre4_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre4_indicator/to_sparse_input/indicesWhere1userGenre4_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre4_indicator/to_sparse_input/valuesGatherNd(userGenre4_indicator/ExpandDims:output:04userGenre4_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre4_indicator/to_sparse_input/dense_shapeShape(userGenre4_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre4_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre4_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre4_indicator/to_sparse_input/values:output:0@usergenre4_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre4_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre4_indicator/SparseToDenseSparseToDense4userGenre4_indicator/to_sparse_input/indices:index:09userGenre4_indicator/to_sparse_input/dense_shape:output:0;userGenre4_indicator/None_Lookup/LookupTableFindV2:values:09userGenre4_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre4_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre4_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre4_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre4_indicator/one_hotOneHot*userGenre4_indicator/SparseToDense:dense:0+userGenre4_indicator/one_hot/depth:output:0+userGenre4_indicator/one_hot/Const:output:0-userGenre4_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre4_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre4_indicator/SumSum%userGenre4_indicator/one_hot:output:03userGenre4_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre4_indicator/ShapeShape!userGenre4_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre4_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre4_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre4_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre4_indicator/strided_sliceStridedSlice#userGenre4_indicator/Shape:output:01userGenre4_indicator/strided_slice/stack:output:03userGenre4_indicator/strided_slice/stack_1:output:03userGenre4_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre4_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre4_indicator/Reshape/shapePack+userGenre4_indicator/strided_slice:output:0-userGenre4_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre4_indicator/ReshapeReshape!userGenre4_indicator/Sum:output:0+userGenre4_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????n
#userGenre5_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userGenre5_indicator/ExpandDims
ExpandDimsfeatures_usergenre5,userGenre5_indicator/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????t
3userGenre5_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-userGenre5_indicator/to_sparse_input/NotEqualNotEqual(userGenre5_indicator/ExpandDims:output:0<userGenre5_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,userGenre5_indicator/to_sparse_input/indicesWhere1userGenre5_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+userGenre5_indicator/to_sparse_input/valuesGatherNd(userGenre5_indicator/ExpandDims:output:04userGenre5_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0userGenre5_indicator/to_sparse_input/dense_shapeShape(userGenre5_indicator/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
2userGenre5_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?usergenre5_indicator_none_lookup_lookuptablefindv2_table_handle4userGenre5_indicator/to_sparse_input/values:output:0@usergenre5_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0userGenre5_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"userGenre5_indicator/SparseToDenseSparseToDense4userGenre5_indicator/to_sparse_input/indices:index:09userGenre5_indicator/to_sparse_input/dense_shape:output:0;userGenre5_indicator/None_Lookup/LookupTableFindV2:values:09userGenre5_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"userGenre5_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$userGenre5_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"userGenre5_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
userGenre5_indicator/one_hotOneHot*userGenre5_indicator/SparseToDense:dense:0+userGenre5_indicator/one_hot/depth:output:0+userGenre5_indicator/one_hot/Const:output:0-userGenre5_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*userGenre5_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
userGenre5_indicator/SumSum%userGenre5_indicator/one_hot:output:03userGenre5_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
userGenre5_indicator/ShapeShape!userGenre5_indicator/Sum:output:0*
T0*
_output_shapes
:r
(userGenre5_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*userGenre5_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*userGenre5_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"userGenre5_indicator/strided_sliceStridedSlice#userGenre5_indicator/Shape:output:01userGenre5_indicator/strided_slice/stack:output:03userGenre5_indicator/strided_slice/stack_1:output:03userGenre5_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$userGenre5_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"userGenre5_indicator/Reshape/shapePack+userGenre5_indicator/strided_slice:output:0-userGenre5_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userGenre5_indicator/ReshapeReshape!userGenre5_indicator/Sum:output:0+userGenre5_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
userRatingCount/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingCount/ExpandDims
ExpandDimsfeatures_userratingcount'userRatingCount/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
userRatingCount/CastCast#userRatingCount/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????]
userRatingCount/ShapeShapeuserRatingCount/Cast:y:0*
T0*
_output_shapes
:m
#userRatingCount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%userRatingCount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%userRatingCount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingCount/strided_sliceStridedSliceuserRatingCount/Shape:output:0,userRatingCount/strided_slice/stack:output:0.userRatingCount/strided_slice/stack_1:output:0.userRatingCount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
userRatingCount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingCount/Reshape/shapePack&userRatingCount/strided_slice:output:0(userRatingCount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingCount/ReshapeReshapeuserRatingCount/Cast:y:0&userRatingCount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
userRatingStddev/ExpandDims
ExpandDimsfeatures_userratingstddev(userRatingStddev/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????j
userRatingStddev/ShapeShape$userRatingStddev/ExpandDims:output:0*
T0*
_output_shapes
:n
$userRatingStddev/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userRatingStddev/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userRatingStddev/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
userRatingStddev/strided_sliceStridedSliceuserRatingStddev/Shape:output:0-userRatingStddev/strided_slice/stack:output:0/userRatingStddev/strided_slice/stack_1:output:0/userRatingStddev/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userRatingStddev/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
userRatingStddev/Reshape/shapePack'userRatingStddev/strided_slice:output:0)userRatingStddev/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
userRatingStddev/ReshapeReshape$userRatingStddev/ExpandDims:output:0'userRatingStddev/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2movieAvgRating/Reshape:output:0&movieGenre1_indicator/Reshape:output:0&movieGenre2_indicator/Reshape:output:0&movieGenre3_indicator/Reshape:output:0!movieRatingCount/Reshape:output:0"movieRatingStddev/Reshape:output:0releaseYear/Reshape:output:0userAvgRating/Reshape:output:0%userGenre1_indicator/Reshape:output:0%userGenre2_indicator/Reshape:output:0%userGenre3_indicator/Reshape:output:0%userGenre4_indicator/Reshape:output:0%userGenre5_indicator/Reshape:output:0 userRatingCount/Reshape:output:0!userRatingStddev/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp4^movieGenre1_indicator/None_Lookup/LookupTableFindV24^movieGenre2_indicator/None_Lookup/LookupTableFindV24^movieGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre1_indicator/None_Lookup/LookupTableFindV23^userGenre2_indicator/None_Lookup/LookupTableFindV23^userGenre3_indicator/None_Lookup/LookupTableFindV23^userGenre4_indicator/None_Lookup/LookupTableFindV23^userGenre5_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : 2j
3movieGenre1_indicator/None_Lookup/LookupTableFindV23movieGenre1_indicator/None_Lookup/LookupTableFindV22j
3movieGenre2_indicator/None_Lookup/LookupTableFindV23movieGenre2_indicator/None_Lookup/LookupTableFindV22j
3movieGenre3_indicator/None_Lookup/LookupTableFindV23movieGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre1_indicator/None_Lookup/LookupTableFindV22userGenre1_indicator/None_Lookup/LookupTableFindV22h
2userGenre2_indicator/None_Lookup/LookupTableFindV22userGenre2_indicator/None_Lookup/LookupTableFindV22h
2userGenre3_indicator/None_Lookup/LookupTableFindV22userGenre3_indicator/None_Lookup/LookupTableFindV22h
2userGenre4_indicator/None_Lookup/LookupTableFindV22userGenre4_indicator/None_Lookup/LookupTableFindV22h
2userGenre5_indicator/None_Lookup/LookupTableFindV22userGenre5_indicator/None_Lookup/LookupTableFindV2:\ X
#
_output_shapes
:?????????
1
_user_specified_namefeatures/movieAvgRating:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre1:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre2:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/movieGenre3:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/movieRatingCount:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/movieRatingStddev:YU
#
_output_shapes
:?????????
.
_user_specified_namefeatures/releaseYear:[W
#
_output_shapes
:?????????
0
_user_specified_namefeatures/userAvgRating:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre1:X	T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre2:X
T
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre3:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre4:XT
#
_output_shapes
:?????????
-
_user_specified_namefeatures/userGenre5:]Y
#
_output_shapes
:?????????
2
_user_specified_namefeatures/userRatingCount:^Z
#
_output_shapes
:?????????
3
_user_specified_namefeatures/userRatingStddev:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_12217
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
__inference__creator_12168
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name558*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
B__inference_fm_layer_layer_call_and_return_conditional_losses_9492
x1
matmul_readvariableop_resource:	?

identity??MatMul/ReadVariableOp?Pow_2/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
PowPowMatMul:product:0Pow/y:output:0*
T0*'
_output_shapes
:?????????
L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_1PowxPow_1/y:output:0*
T0*(
_output_shapes
:??????????t
Pow_2/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
Pow_2PowPow_2/ReadVariableOp:value:0Pow_2/y:output:0*
T0*
_output_shapes
:	?
Z
MatMul_1MatMul	Pow_1:z:0	Pow_2:z:0*
T0*'
_output_shapes
:?????????
Y
subSubPow:z:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
SumSumsub:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Z
mulMulSum:output:0mul/y:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentitymul:z:0^NoOp*
T0*'
_output_shapes
:?????????u
NoOpNoOp^MatMul/ReadVariableOp^Pow_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2,
Pow_2/ReadVariableOpPow_2/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
:
__inference__creator_12114
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name408*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_122332
.table_init371_lookuptableimportv2_table_handle*
&table_init371_lookuptableimportv2_keys,
(table_init371_lookuptableimportv2_values	
identity??!table_init371/LookupTableImportV2?
!table_init371/LookupTableImportV2LookupTableImportV2.table_init371_lookuptableimportv2_table_handle&table_init371_lookuptableimportv2_keys(table_init371_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init371/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init371/LookupTableImportV2!table_init371/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?9
?
?__inference_model_layer_call_and_return_conditional_losses_9524

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
dense_features_9411
dense_features_9413	
dense_features_9415
dense_features_9417	
dense_features_9419
dense_features_9421	
dense_features_9423
dense_features_9425	
dense_features_9427
dense_features_9429	
dense_features_9431
dense_features_9433	
dense_features_9435
dense_features_9437	
dense_features_9439
dense_features_9441	

dense_9467:	?

dense_9469: 
fm_layer_9493:	?

identity??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?&dense_features/StatefulPartitionedCall? fm_layer/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14dense_features_9411dense_features_9413dense_features_9415dense_features_9417dense_features_9419dense_features_9421dense_features_9423dense_features_9425dense_features_9427dense_features_9429dense_features_9431dense_features_9433dense_features_9435dense_features_9437dense_features_9439dense_features_9441**
Tin#
!2								*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_features_layer_call_and_return_conditional_losses_9410?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0
dense_9467
dense_9469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_9466?
 fm_layer/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0fm_layer_9493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_fm_layer_layer_call_and_return_conditional_losses_9492?
add/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0)fm_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_9502?
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_9509w
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp
dense_9467*
_output_shapes
:	?*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp
dense_9469*
_output_shapes
:*
dtype0?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp'^dense_features/StatefulPartitionedCall!^fm_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2D
 fm_layer/StatefulPartitionedCall fm_layer/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:K	G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:K
G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_121942
.table_init593_lookuptableimportv2_table_handle*
&table_init593_lookuptableimportv2_keys,
(table_init593_lookuptableimportv2_values	
identity??!table_init593/LookupTableImportV2?
!table_init593/LookupTableImportV2LookupTableImportV2.table_init593_lookuptableimportv2_table_handle&table_init593_lookuptableimportv2_keys(table_init593_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init593/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init593/LookupTableImportV2!table_init593/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
:
__inference__creator_12186
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name594*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_122252
.table_init335_lookuptableimportv2_table_handle*
&table_init335_lookuptableimportv2_keys,
(table_init335_lookuptableimportv2_values	
identity??!table_init335/LookupTableImportV2?
!table_init335/LookupTableImportV2LookupTableImportV2.table_init335_lookuptableimportv2_table_handle&table_init335_lookuptableimportv2_keys(table_init335_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init335/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init335/LookupTableImportV2!table_init335/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
g
=__inference_add_layer_call_and_return_conditional_losses_9502

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_122732
.table_init593_lookuptableimportv2_table_handle*
&table_init593_lookuptableimportv2_keys,
(table_init593_lookuptableimportv2_values	
identity??!table_init593/LookupTableImportV2?
!table_init593/LookupTableImportV2LookupTableImportV2.table_init593_lookuptableimportv2_table_handle&table_init593_lookuptableimportv2_keys(table_init593_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init593/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init593/LookupTableImportV2!table_init593/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:"?M
saver_filename:0StatefulPartitionedCall_9:0StatefulPartitionedCall_108"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
movieAvgRating3
 serving_default_movieAvgRating:0?????????
?
movieGenre10
serving_default_movieGenre1:0?????????
?
movieGenre20
serving_default_movieGenre2:0?????????
?
movieGenre30
serving_default_movieGenre3:0?????????
I
movieRatingCount5
"serving_default_movieRatingCount:0?????????
K
movieRatingStddev6
#serving_default_movieRatingStddev:0?????????
?
releaseYear0
serving_default_releaseYear:0?????????
C
userAvgRating2
serving_default_userAvgRating:0?????????
=

userGenre1/
serving_default_userGenre1:0?????????
=

userGenre2/
serving_default_userGenre2:0?????????
=

userGenre3/
serving_default_userGenre3:0?????????
=

userGenre4/
serving_default_userGenre4:0?????????
=

userGenre5/
serving_default_userGenre5:0?????????
G
userRatingCount4
!serving_default_userRatingCount:0?????????
I
userRatingStddev5
"serving_default_userRatingStddev:0?????????@

activation2
StatefulPartitionedCall_8:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-0
layer-16
layer_with_weights-1
layer-17
layer-18
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_feature_columns
%
_resources"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
5
,0
-1
42"
trackable_list_wrapper
5
,0
-1
42"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32?
$__inference_model_layer_call_fn_9565
%__inference_model_layer_call_fn_10536
%__inference_model_layer_call_fn_10593
%__inference_model_layer_call_fn_10244?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
?
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32?
@__inference_model_layer_call_and_return_conditional_losses_10930
@__inference_model_layer_call_and_return_conditional_losses_11267
@__inference_model_layer_call_and_return_conditional_losses_10317
@__inference_model_layer_call_and_return_conditional_losses_10390?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
?B?
__inference__wrapped_model_9074movieAvgRatingmovieGenre1movieGenre2movieGenre3movieRatingCountmovieRatingStddevreleaseYearuserAvgRating
userGenre1
userGenre2
userGenre3
userGenre4
userGenre5userRatingCountuserRatingStddev"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rate,m?-m?4m?,v?-v?4v?"
	optimizer
,
Userving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?
[trace_0
\trace_12?
.__inference_dense_features_layer_call_fn_11318
.__inference_dense_features_layer_call_fn_11369?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z[trace_0z\trace_1
?
]trace_0
^trace_12?
I__inference_dense_features_layer_call_and_return_conditional_losses_11670
I__inference_dense_features_layer_call_and_return_conditional_losses_11971?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z]trace_0z^trace_1
 "
trackable_list_wrapper
?
_movieGenre1
`movieGenre2
amovieGenre3
b
userGenre1
c
userGenre2
d
userGenre3
e
userGenre4
f
userGenre5"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
ltrace_02?
%__inference_dense_layer_call_fn_11980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zltrace_0
?
mtrace_02?
@__inference_dense_layer_call_and_return_conditional_losses_12002?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zmtrace_0
:	?2dense/kernel
:2
dense/bias
'
40"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?
strace_02?
(__inference_fm_layer_layer_call_fn_12009?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zstrace_0
?
ttrace_02?
C__inference_fm_layer_layer_call_and_return_conditional_losses_12029?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zttrace_0
": 	?
2fm_layer/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?
ztrace_02?
#__inference_add_layer_call_fn_12035?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zztrace_0
?
{trace_02?
>__inference_add_layer_call_and_return_conditional_losses_12041?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z{trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_activation_layer_call_fn_12046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_activation_layer_call_and_return_conditional_losses_12051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
__inference_loss_fn_0_12062?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference_loss_fn_1_12073?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_model_layer_call_fn_9565movieAvgRatingmovieGenre1movieGenre2movieGenre3movieRatingCountmovieRatingStddevreleaseYearuserAvgRating
userGenre1
userGenre2
userGenre3
userGenre4
userGenre5userRatingCountuserRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_10536inputs/movieAvgRatinginputs/movieGenre1inputs/movieGenre2inputs/movieGenre3inputs/movieRatingCountinputs/movieRatingStddevinputs/releaseYearinputs/userAvgRatinginputs/userGenre1inputs/userGenre2inputs/userGenre3inputs/userGenre4inputs/userGenre5inputs/userRatingCountinputs/userRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_10593inputs/movieAvgRatinginputs/movieGenre1inputs/movieGenre2inputs/movieGenre3inputs/movieRatingCountinputs/movieRatingStddevinputs/releaseYearinputs/userAvgRatinginputs/userGenre1inputs/userGenre2inputs/userGenre3inputs/userGenre4inputs/userGenre5inputs/userRatingCountinputs/userRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_10244movieAvgRatingmovieGenre1movieGenre2movieGenre3movieRatingCountmovieRatingStddevreleaseYearuserAvgRating
userGenre1
userGenre2
userGenre3
userGenre4
userGenre5userRatingCountuserRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_10930inputs/movieAvgRatinginputs/movieGenre1inputs/movieGenre2inputs/movieGenre3inputs/movieRatingCountinputs/movieRatingStddevinputs/releaseYearinputs/userAvgRatinginputs/userGenre1inputs/userGenre2inputs/userGenre3inputs/userGenre4inputs/userGenre5inputs/userRatingCountinputs/userRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_11267inputs/movieAvgRatinginputs/movieGenre1inputs/movieGenre2inputs/movieGenre3inputs/movieRatingCountinputs/movieRatingStddevinputs/releaseYearinputs/userAvgRatinginputs/userGenre1inputs/userGenre2inputs/userGenre3inputs/userGenre4inputs/userGenre5inputs/userRatingCountinputs/userRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_10317movieAvgRatingmovieGenre1movieGenre2movieGenre3movieRatingCountmovieRatingStddevreleaseYearuserAvgRating
userGenre1
userGenre2
userGenre3
userGenre4
userGenre5userRatingCountuserRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_10390movieAvgRatingmovieGenre1movieGenre2movieGenre3movieRatingCountmovieRatingStddevreleaseYearuserAvgRating
userGenre1
userGenre2
userGenre3
userGenre4
userGenre5userRatingCountuserRatingStddev"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_10467movieAvgRatingmovieGenre1movieGenre2movieGenre3movieRatingCountmovieRatingStddevreleaseYearuserAvgRating
userGenre1
userGenre2
userGenre3
userGenre4
userGenre5userRatingCountuserRatingStddev"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_dense_features_layer_call_fn_11318features/movieAvgRatingfeatures/movieGenre1features/movieGenre2features/movieGenre3features/movieRatingCountfeatures/movieRatingStddevfeatures/releaseYearfeatures/userAvgRatingfeatures/userGenre1features/userGenre2features/userGenre3features/userGenre4features/userGenre5features/userRatingCountfeatures/userRatingStddev"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_dense_features_layer_call_fn_11369features/movieAvgRatingfeatures/movieGenre1features/movieGenre2features/movieGenre3features/movieRatingCountfeatures/movieRatingStddevfeatures/releaseYearfeatures/userAvgRatingfeatures/userGenre1features/userGenre2features/userGenre3features/userGenre4features/userGenre5features/userRatingCountfeatures/userRatingStddev"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_dense_features_layer_call_and_return_conditional_losses_11670features/movieAvgRatingfeatures/movieGenre1features/movieGenre2features/movieGenre3features/movieRatingCountfeatures/movieRatingStddevfeatures/releaseYearfeatures/userAvgRatingfeatures/userGenre1features/userGenre2features/userGenre3features/userGenre4features/userGenre5features/userRatingCountfeatures/userRatingStddev"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_dense_features_layer_call_and_return_conditional_losses_11971features/movieAvgRatingfeatures/movieGenre1features/movieGenre2features/movieGenre3features/movieRatingCountfeatures/movieRatingStddevfeatures/releaseYearfeatures/userAvgRatingfeatures/userGenre1features/userGenre2features/userGenre3features/userGenre4features/userGenre5features/userRatingCountfeatures/userRatingStddev"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
7
?movieGenre1_lookup"
_generic_user_object
7
?movieGenre2_lookup"
_generic_user_object
7
?movieGenre3_lookup"
_generic_user_object
6
?userGenre1_lookup"
_generic_user_object
6
?userGenre2_lookup"
_generic_user_object
6
?userGenre3_lookup"
_generic_user_object
6
?userGenre4_lookup"
_generic_user_object
6
?userGenre5_lookup"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_dense_layer_call_fn_11980inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_12002inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_fm_layer_layer_call_fn_12009x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_fm_layer_layer_call_and_return_conditional_losses_12029x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_add_layer_call_fn_12035inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
>__inference_add_layer_call_and_return_conditional_losses_12041inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_activation_layer_call_fn_12046inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_activation_layer_call_and_return_conditional_losses_12051inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_loss_fn_0_12062"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_loss_fn_1_12073"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
?
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives"
_tf_keras_metric
?
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives"
_tf_keras_metric
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
"
_generic_user_object
?
?trace_02?
__inference__creator_12078?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12086?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12091?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12096?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12104?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12109?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12114?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12122?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12127?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12132?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12140?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12145?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12150?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12158?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12163?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12168?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12176?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12181?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12186?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12194?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12199?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_12204?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_12212?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_12217?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?B?
__inference__creator_12078"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12086"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12091"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12096"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12104"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12109"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12114"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12122"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12127"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12132"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12140"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12145"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12150"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12158"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12163"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12168"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12176"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12181"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12186"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12194"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12199"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_12204"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_12212"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_12217"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
$:"	?2Adam/dense/kernel/m
:2Adam/dense/bias/m
':%	?
2Adam/fm_layer/kernel/m
$:"	?2Adam/dense/kernel/v
:2Adam/dense/bias/v
':%	?
2Adam/fm_layer/kernel/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_15jtf.TrackableConstant
"J

Const_16jtf.TrackableConstant
"J

Const_17jtf.TrackableConstant
"J

Const_18jtf.TrackableConstant
"J

Const_19jtf.TrackableConstant
"J

Const_20jtf.TrackableConstant
"J

Const_21jtf.TrackableConstant
"J

Const_22jtf.TrackableConstant
"J

Const_23jtf.TrackableConstant6
__inference__creator_12078?

? 
? "? 6
__inference__creator_12096?

? 
? "? 6
__inference__creator_12114?

? 
? "? 6
__inference__creator_12132?

? 
? "? 6
__inference__creator_12150?

? 
? "? 6
__inference__creator_12168?

? 
? "? 6
__inference__creator_12186?

? 
? "? 6
__inference__creator_12204?

? 
? "? 8
__inference__destroyer_12091?

? 
? "? 8
__inference__destroyer_12109?

? 
? "? 8
__inference__destroyer_12127?

? 
? "? 8
__inference__destroyer_12145?

? 
? "? 8
__inference__destroyer_12163?

? 
? "? 8
__inference__destroyer_12181?

? 
? "? 8
__inference__destroyer_12199?

? 
? "? 8
__inference__destroyer_12217?

? 
? "? B
__inference__initializer_12086 ????

? 
? "? B
__inference__initializer_12104 ????

? 
? "? B
__inference__initializer_12122 ????

? 
? "? B
__inference__initializer_12140 ????

? 
? "? B
__inference__initializer_12158 ????

? 
? "? B
__inference__initializer_12176 ????

? 
? "? B
__inference__initializer_12194 ????

? 
? "? B
__inference__initializer_12212 ????

? 
? "? ?
__inference__wrapped_model_9074?#????????????????,-4???
???
???
6
movieAvgRating$?!
movieAvgRating?????????
0
movieGenre1!?
movieGenre1?????????
0
movieGenre2!?
movieGenre2?????????
0
movieGenre3!?
movieGenre3?????????
:
movieRatingCount&?#
movieRatingCount?????????
<
movieRatingStddev'?$
movieRatingStddev?????????
0
releaseYear!?
releaseYear?????????
4
userAvgRating#? 
userAvgRating?????????
.

userGenre1 ?

userGenre1?????????
.

userGenre2 ?

userGenre2?????????
.

userGenre3 ?

userGenre3?????????
.

userGenre4 ?

userGenre4?????????
.

userGenre5 ?

userGenre5?????????
8
userRatingCount%?"
userRatingCount?????????
:
userRatingStddev&?#
userRatingStddev?????????
? "7?4
2

activation$?!

activation??????????
E__inference_activation_layer_call_and_return_conditional_losses_12051X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_activation_layer_call_fn_12046K/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_add_layer_call_and_return_conditional_losses_12041?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
#__inference_add_layer_call_fn_12035vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
I__inference_dense_features_layer_call_and_return_conditional_losses_11670? ???????????????????
???
???
?
movieAvgRating-?*
features/movieAvgRating?????????
9
movieGenre1*?'
features/movieGenre1?????????
9
movieGenre2*?'
features/movieGenre2?????????
9
movieGenre3*?'
features/movieGenre3?????????
C
movieRatingCount/?,
features/movieRatingCount?????????
E
movieRatingStddev0?-
features/movieRatingStddev?????????
9
releaseYear*?'
features/releaseYear?????????
=
userAvgRating,?)
features/userAvgRating?????????
7

userGenre1)?&
features/userGenre1?????????
7

userGenre2)?&
features/userGenre2?????????
7

userGenre3)?&
features/userGenre3?????????
7

userGenre4)?&
features/userGenre4?????????
7

userGenre5)?&
features/userGenre5?????????
A
userRatingCount.?+
features/userRatingCount?????????
C
userRatingStddev/?,
features/userRatingStddev?????????

 
p 
? "&?#
?
0??????????
? ?
I__inference_dense_features_layer_call_and_return_conditional_losses_11971? ???????????????????
???
???
?
movieAvgRating-?*
features/movieAvgRating?????????
9
movieGenre1*?'
features/movieGenre1?????????
9
movieGenre2*?'
features/movieGenre2?????????
9
movieGenre3*?'
features/movieGenre3?????????
C
movieRatingCount/?,
features/movieRatingCount?????????
E
movieRatingStddev0?-
features/movieRatingStddev?????????
9
releaseYear*?'
features/releaseYear?????????
=
userAvgRating,?)
features/userAvgRating?????????
7

userGenre1)?&
features/userGenre1?????????
7

userGenre2)?&
features/userGenre2?????????
7

userGenre3)?&
features/userGenre3?????????
7

userGenre4)?&
features/userGenre4?????????
7

userGenre5)?&
features/userGenre5?????????
A
userRatingCount.?+
features/userRatingCount?????????
C
userRatingStddev/?,
features/userRatingStddev?????????

 
p
? "&?#
?
0??????????
? ?
.__inference_dense_features_layer_call_fn_11318? ???????????????????
???
???
?
movieAvgRating-?*
features/movieAvgRating?????????
9
movieGenre1*?'
features/movieGenre1?????????
9
movieGenre2*?'
features/movieGenre2?????????
9
movieGenre3*?'
features/movieGenre3?????????
C
movieRatingCount/?,
features/movieRatingCount?????????
E
movieRatingStddev0?-
features/movieRatingStddev?????????
9
releaseYear*?'
features/releaseYear?????????
=
userAvgRating,?)
features/userAvgRating?????????
7

userGenre1)?&
features/userGenre1?????????
7

userGenre2)?&
features/userGenre2?????????
7

userGenre3)?&
features/userGenre3?????????
7

userGenre4)?&
features/userGenre4?????????
7

userGenre5)?&
features/userGenre5?????????
A
userRatingCount.?+
features/userRatingCount?????????
C
userRatingStddev/?,
features/userRatingStddev?????????

 
p 
? "????????????
.__inference_dense_features_layer_call_fn_11369? ???????????????????
???
???
?
movieAvgRating-?*
features/movieAvgRating?????????
9
movieGenre1*?'
features/movieGenre1?????????
9
movieGenre2*?'
features/movieGenre2?????????
9
movieGenre3*?'
features/movieGenre3?????????
C
movieRatingCount/?,
features/movieRatingCount?????????
E
movieRatingStddev0?-
features/movieRatingStddev?????????
9
releaseYear*?'
features/releaseYear?????????
=
userAvgRating,?)
features/userAvgRating?????????
7

userGenre1)?&
features/userGenre1?????????
7

userGenre2)?&
features/userGenre2?????????
7

userGenre3)?&
features/userGenre3?????????
7

userGenre4)?&
features/userGenre4?????????
7

userGenre5)?&
features/userGenre5?????????
A
userRatingCount.?+
features/userRatingCount?????????
C
userRatingStddev/?,
features/userRatingStddev?????????

 
p
? "????????????
@__inference_dense_layer_call_and_return_conditional_losses_12002],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
%__inference_dense_layer_call_fn_11980P,-0?-
&?#
!?
inputs??????????
? "???????????
C__inference_fm_layer_layer_call_and_return_conditional_losses_12029W4+?(
!?
?
x??????????
? "%?"
?
0?????????
? v
(__inference_fm_layer_layer_call_fn_12009J4+?(
!?
?
x??????????
? "??????????:
__inference_loss_fn_0_12062,?

? 
? "? :
__inference_loss_fn_1_12073-?

? 
? "? ?
@__inference_model_layer_call_and_return_conditional_losses_10317?#????????????????,-4???
???
???
6
movieAvgRating$?!
movieAvgRating?????????
0
movieGenre1!?
movieGenre1?????????
0
movieGenre2!?
movieGenre2?????????
0
movieGenre3!?
movieGenre3?????????
:
movieRatingCount&?#
movieRatingCount?????????
<
movieRatingStddev'?$
movieRatingStddev?????????
0
releaseYear!?
releaseYear?????????
4
userAvgRating#? 
userAvgRating?????????
.

userGenre1 ?

userGenre1?????????
.

userGenre2 ?

userGenre2?????????
.

userGenre3 ?

userGenre3?????????
.

userGenre4 ?

userGenre4?????????
.

userGenre5 ?

userGenre5?????????
8
userRatingCount%?"
userRatingCount?????????
:
userRatingStddev&?#
userRatingStddev?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_10390?#????????????????,-4???
???
???
6
movieAvgRating$?!
movieAvgRating?????????
0
movieGenre1!?
movieGenre1?????????
0
movieGenre2!?
movieGenre2?????????
0
movieGenre3!?
movieGenre3?????????
:
movieRatingCount&?#
movieRatingCount?????????
<
movieRatingStddev'?$
movieRatingStddev?????????
0
releaseYear!?
releaseYear?????????
4
userAvgRating#? 
userAvgRating?????????
.

userGenre1 ?

userGenre1?????????
.

userGenre2 ?

userGenre2?????????
.

userGenre3 ?

userGenre3?????????
.

userGenre4 ?

userGenre4?????????
.

userGenre5 ?

userGenre5?????????
8
userRatingCount%?"
userRatingCount?????????
:
userRatingStddev&?#
userRatingStddev?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_10930?#????????????????,-4???
???
???
=
movieAvgRating+?(
inputs/movieAvgRating?????????
7
movieGenre1(?%
inputs/movieGenre1?????????
7
movieGenre2(?%
inputs/movieGenre2?????????
7
movieGenre3(?%
inputs/movieGenre3?????????
A
movieRatingCount-?*
inputs/movieRatingCount?????????
C
movieRatingStddev.?+
inputs/movieRatingStddev?????????
7
releaseYear(?%
inputs/releaseYear?????????
;
userAvgRating*?'
inputs/userAvgRating?????????
5

userGenre1'?$
inputs/userGenre1?????????
5

userGenre2'?$
inputs/userGenre2?????????
5

userGenre3'?$
inputs/userGenre3?????????
5

userGenre4'?$
inputs/userGenre4?????????
5

userGenre5'?$
inputs/userGenre5?????????
?
userRatingCount,?)
inputs/userRatingCount?????????
A
userRatingStddev-?*
inputs/userRatingStddev?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_11267?#????????????????,-4???
???
???
=
movieAvgRating+?(
inputs/movieAvgRating?????????
7
movieGenre1(?%
inputs/movieGenre1?????????
7
movieGenre2(?%
inputs/movieGenre2?????????
7
movieGenre3(?%
inputs/movieGenre3?????????
A
movieRatingCount-?*
inputs/movieRatingCount?????????
C
movieRatingStddev.?+
inputs/movieRatingStddev?????????
7
releaseYear(?%
inputs/releaseYear?????????
;
userAvgRating*?'
inputs/userAvgRating?????????
5

userGenre1'?$
inputs/userGenre1?????????
5

userGenre2'?$
inputs/userGenre2?????????
5

userGenre3'?$
inputs/userGenre3?????????
5

userGenre4'?$
inputs/userGenre4?????????
5

userGenre5'?$
inputs/userGenre5?????????
?
userRatingCount,?)
inputs/userRatingCount?????????
A
userRatingStddev-?*
inputs/userRatingStddev?????????
p

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_10244?#????????????????,-4???
???
???
6
movieAvgRating$?!
movieAvgRating?????????
0
movieGenre1!?
movieGenre1?????????
0
movieGenre2!?
movieGenre2?????????
0
movieGenre3!?
movieGenre3?????????
:
movieRatingCount&?#
movieRatingCount?????????
<
movieRatingStddev'?$
movieRatingStddev?????????
0
releaseYear!?
releaseYear?????????
4
userAvgRating#? 
userAvgRating?????????
.

userGenre1 ?

userGenre1?????????
.

userGenre2 ?

userGenre2?????????
.

userGenre3 ?

userGenre3?????????
.

userGenre4 ?

userGenre4?????????
.

userGenre5 ?

userGenre5?????????
8
userRatingCount%?"
userRatingCount?????????
:
userRatingStddev&?#
userRatingStddev?????????
p

 
? "???????????
%__inference_model_layer_call_fn_10536?#????????????????,-4???
???
???
=
movieAvgRating+?(
inputs/movieAvgRating?????????
7
movieGenre1(?%
inputs/movieGenre1?????????
7
movieGenre2(?%
inputs/movieGenre2?????????
7
movieGenre3(?%
inputs/movieGenre3?????????
A
movieRatingCount-?*
inputs/movieRatingCount?????????
C
movieRatingStddev.?+
inputs/movieRatingStddev?????????
7
releaseYear(?%
inputs/releaseYear?????????
;
userAvgRating*?'
inputs/userAvgRating?????????
5

userGenre1'?$
inputs/userGenre1?????????
5

userGenre2'?$
inputs/userGenre2?????????
5

userGenre3'?$
inputs/userGenre3?????????
5

userGenre4'?$
inputs/userGenre4?????????
5

userGenre5'?$
inputs/userGenre5?????????
?
userRatingCount,?)
inputs/userRatingCount?????????
A
userRatingStddev-?*
inputs/userRatingStddev?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_10593?#????????????????,-4???
???
???
=
movieAvgRating+?(
inputs/movieAvgRating?????????
7
movieGenre1(?%
inputs/movieGenre1?????????
7
movieGenre2(?%
inputs/movieGenre2?????????
7
movieGenre3(?%
inputs/movieGenre3?????????
A
movieRatingCount-?*
inputs/movieRatingCount?????????
C
movieRatingStddev.?+
inputs/movieRatingStddev?????????
7
releaseYear(?%
inputs/releaseYear?????????
;
userAvgRating*?'
inputs/userAvgRating?????????
5

userGenre1'?$
inputs/userGenre1?????????
5

userGenre2'?$
inputs/userGenre2?????????
5

userGenre3'?$
inputs/userGenre3?????????
5

userGenre4'?$
inputs/userGenre4?????????
5

userGenre5'?$
inputs/userGenre5?????????
?
userRatingCount,?)
inputs/userRatingCount?????????
A
userRatingStddev-?*
inputs/userRatingStddev?????????
p

 
? "???????????
$__inference_model_layer_call_fn_9565?#????????????????,-4???
???
???
6
movieAvgRating$?!
movieAvgRating?????????
0
movieGenre1!?
movieGenre1?????????
0
movieGenre2!?
movieGenre2?????????
0
movieGenre3!?
movieGenre3?????????
:
movieRatingCount&?#
movieRatingCount?????????
<
movieRatingStddev'?$
movieRatingStddev?????????
0
releaseYear!?
releaseYear?????????
4
userAvgRating#? 
userAvgRating?????????
.

userGenre1 ?

userGenre1?????????
.

userGenre2 ?

userGenre2?????????
.

userGenre3 ?

userGenre3?????????
.

userGenre4 ?

userGenre4?????????
.

userGenre5 ?

userGenre5?????????
8
userRatingCount%?"
userRatingCount?????????
:
userRatingStddev&?#
userRatingStddev?????????
p 

 
? "???????????
#__inference_signature_wrapper_10467?#????????????????,-4???
? 
???
6
movieAvgRating$?!
movieAvgRating?????????
0
movieGenre1!?
movieGenre1?????????
0
movieGenre2!?
movieGenre2?????????
0
movieGenre3!?
movieGenre3?????????
:
movieRatingCount&?#
movieRatingCount?????????
<
movieRatingStddev'?$
movieRatingStddev?????????
0
releaseYear!?
releaseYear?????????
4
userAvgRating#? 
userAvgRating?????????
.

userGenre1 ?

userGenre1?????????
.

userGenre2 ?

userGenre2?????????
.

userGenre3 ?

userGenre3?????????
.

userGenre4 ?

userGenre4?????????
.

userGenre5 ?

userGenre5?????????
8
userRatingCount%?"
userRatingCount?????????
:
userRatingStddev&?#
userRatingStddev?????????"7?4
2

activation$?!

activation?????????