??$
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
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

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
?
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
Ttype"
Indextype:
2	"

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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:n
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:n
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
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
?
$tcn/residual_block_0/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_0/conv1D_0/kernel
?
8tcn/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_0/kernel*"
_output_shapes
:n*
dtype0
?
"tcn/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_0/conv1D_0/bias
?
6tcn/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_0/conv1D_0/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_0/conv1D_1/kernel
?
8tcn/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_1/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_0/conv1D_1/bias
?
6tcn/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_0/conv1D_1/bias*
_output_shapes
:n*
dtype0
?
+tcn/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*<
shared_name-+tcn/residual_block_0/matching_conv1D/kernel
?
?tcn/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp+tcn/residual_block_0/matching_conv1D/kernel*"
_output_shapes
:n*
dtype0
?
)tcn/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*:
shared_name+)tcn/residual_block_0/matching_conv1D/bias
?
=tcn/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp)tcn/residual_block_0/matching_conv1D/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_1/conv1D_0/kernel
?
8tcn/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_0/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_1/conv1D_0/bias
?
6tcn/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_1/conv1D_0/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_1/conv1D_1/kernel
?
8tcn/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_1/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_1/conv1D_1/bias
?
6tcn/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_1/conv1D_1/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_2/conv1D_0/kernel
?
8tcn/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_0/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_2/conv1D_0/bias
?
6tcn/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_2/conv1D_0/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_2/conv1D_1/kernel
?
8tcn/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_1/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_2/conv1D_1/bias
?
6tcn/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_2/conv1D_1/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_3/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_3/conv1D_0/kernel
?
8tcn/residual_block_3/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_0/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_3/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_3/conv1D_0/bias
?
6tcn/residual_block_3/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_3/conv1D_0/bias*
_output_shapes
:n*
dtype0
?
$tcn/residual_block_3/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*5
shared_name&$tcn/residual_block_3/conv1D_1/kernel
?
8tcn/residual_block_3/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_1/kernel*"
_output_shapes
:nn*
dtype0
?
"tcn/residual_block_3/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*3
shared_name$"tcn/residual_block_3/conv1D_1/bias
?
6tcn/residual_block_3/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_3/conv1D_1/bias*
_output_shapes
:n*
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
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
x
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:n
*
shared_namedense/kernel/m
q
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes

:n
*
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
:
*
dtype0
?
&tcn/residual_block_0/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*7
shared_name(&tcn/residual_block_0/conv1D_0/kernel/m
?
:tcn/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_0/conv1D_0/kernel/m*"
_output_shapes
:n*
dtype0
?
$tcn/residual_block_0/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_0/conv1D_0/bias/m
?
8tcn/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_0/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_0/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_0/conv1D_1/kernel/m
?
:tcn/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_0/conv1D_1/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_0/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_0/conv1D_1/bias/m
?
8tcn/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_1/bias/m*
_output_shapes
:n*
dtype0
?
-tcn/residual_block_0/matching_conv1D/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*>
shared_name/-tcn/residual_block_0/matching_conv1D/kernel/m
?
Atcn/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpReadVariableOp-tcn/residual_block_0/matching_conv1D/kernel/m*"
_output_shapes
:n*
dtype0
?
+tcn/residual_block_0/matching_conv1D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*<
shared_name-+tcn/residual_block_0/matching_conv1D/bias/m
?
?tcn/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOpReadVariableOp+tcn/residual_block_0/matching_conv1D/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_1/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_1/conv1D_0/kernel/m
?
:tcn/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_1/conv1D_0/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_1/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_1/conv1D_0/bias/m
?
8tcn/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_0/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_1/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_1/conv1D_1/kernel/m
?
:tcn/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_1/conv1D_1/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_1/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_1/conv1D_1/bias/m
?
8tcn/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_1/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_2/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_2/conv1D_0/kernel/m
?
:tcn/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_2/conv1D_0/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_2/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_2/conv1D_0/bias/m
?
8tcn/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_0/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_2/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_2/conv1D_1/kernel/m
?
:tcn/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_2/conv1D_1/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_2/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_2/conv1D_1/bias/m
?
8tcn/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_1/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_3/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_3/conv1D_0/kernel/m
?
:tcn/residual_block_3/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_3/conv1D_0/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_3/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_3/conv1D_0/bias/m
?
8tcn/residual_block_3/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_0/bias/m*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_3/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_3/conv1D_1/kernel/m
?
:tcn/residual_block_3/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp&tcn/residual_block_3/conv1D_1/kernel/m*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_3/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_3/conv1D_1/bias/m
?
8tcn/residual_block_3/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_1/bias/m*
_output_shapes
:n*
dtype0
x
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:n
*
shared_namedense/kernel/v
q
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes

:n
*
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
:
*
dtype0
?
&tcn/residual_block_0/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*7
shared_name(&tcn/residual_block_0/conv1D_0/kernel/v
?
:tcn/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_0/conv1D_0/kernel/v*"
_output_shapes
:n*
dtype0
?
$tcn/residual_block_0/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_0/conv1D_0/bias/v
?
8tcn/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_0/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_0/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_0/conv1D_1/kernel/v
?
:tcn/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_0/conv1D_1/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_0/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_0/conv1D_1/bias/v
?
8tcn/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_1/bias/v*
_output_shapes
:n*
dtype0
?
-tcn/residual_block_0/matching_conv1D/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*>
shared_name/-tcn/residual_block_0/matching_conv1D/kernel/v
?
Atcn/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpReadVariableOp-tcn/residual_block_0/matching_conv1D/kernel/v*"
_output_shapes
:n*
dtype0
?
+tcn/residual_block_0/matching_conv1D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*<
shared_name-+tcn/residual_block_0/matching_conv1D/bias/v
?
?tcn/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOpReadVariableOp+tcn/residual_block_0/matching_conv1D/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_1/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_1/conv1D_0/kernel/v
?
:tcn/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_1/conv1D_0/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_1/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_1/conv1D_0/bias/v
?
8tcn/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_0/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_1/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_1/conv1D_1/kernel/v
?
:tcn/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_1/conv1D_1/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_1/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_1/conv1D_1/bias/v
?
8tcn/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_1/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_2/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_2/conv1D_0/kernel/v
?
:tcn/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_2/conv1D_0/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_2/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_2/conv1D_0/bias/v
?
8tcn/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_0/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_2/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_2/conv1D_1/kernel/v
?
:tcn/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_2/conv1D_1/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_2/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_2/conv1D_1/bias/v
?
8tcn/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_1/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_3/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_3/conv1D_0/kernel/v
?
:tcn/residual_block_3/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_3/conv1D_0/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_3/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_3/conv1D_0/bias/v
?
8tcn/residual_block_3/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_0/bias/v*
_output_shapes
:n*
dtype0
?
&tcn/residual_block_3/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:nn*7
shared_name(&tcn/residual_block_3/conv1D_1/kernel/v
?
:tcn/residual_block_3/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp&tcn/residual_block_3/conv1D_1/kernel/v*"
_output_shapes
:nn*
dtype0
?
$tcn/residual_block_3/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:n*5
shared_name&$tcn/residual_block_3/conv1D_1/bias/v
?
8tcn/residual_block_3/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_1/bias/v*
_output_shapes
:n*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
		dilations

skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
residual_block_3
slicer_layer
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
 learning_ratem?m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?v?v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
18
19
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
18
19
 
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 

0
1
2
3
 
?

8layers
9shape_match_conv
:final_activation
;conv1D_0
<Act_Conv1D_0
=
SDropout_0
>conv1D_1
?Act_Conv1D_1
@
SDropout_1
AAct_Conv_Blocks
9matching_conv1D
:Act_Res_Block
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?

Flayers
Gshape_match_conv
Hfinal_activation
Iconv1D_0
JAct_Conv1D_0
K
SDropout_0
Lconv1D_1
MAct_Conv1D_1
N
SDropout_1
OAct_Conv_Blocks
Gmatching_identity
HAct_Res_Block
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?

Tlayers
Ushape_match_conv
Vfinal_activation
Wconv1D_0
XAct_Conv1D_0
Y
SDropout_0
Zconv1D_1
[Act_Conv1D_1
\
SDropout_1
]Act_Conv_Blocks
Umatching_identity
VAct_Res_Block
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?

blayers
cshape_match_conv
dfinal_activation
econv1D_0
fAct_Conv1D_0
g
SDropout_0
hconv1D_1
iAct_Conv1D_1
j
SDropout_1
kAct_Conv_Blocks
cmatching_identity
dAct_Res_Block
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
 
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$tcn/residual_block_0/conv1D_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"tcn/residual_block_0/conv1D_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$tcn/residual_block_0/conv1D_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"tcn/residual_block_0/conv1D_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+tcn/residual_block_0/matching_conv1D/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)tcn/residual_block_0/matching_conv1D/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$tcn/residual_block_1/conv1D_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"tcn/residual_block_1/conv1D_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$tcn/residual_block_1/conv1D_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"tcn/residual_block_1/conv1D_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$tcn/residual_block_2/conv1D_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"tcn/residual_block_2/conv1D_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$tcn/residual_block_2/conv1D_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"tcn/residual_block_2/conv1D_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$tcn/residual_block_3/conv1D_0/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"tcn/residual_block_3/conv1D_0/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$tcn/residual_block_3/conv1D_1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"tcn/residual_block_3/conv1D_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

~0
1
?2
 
 
1
;0
<1
=2
>3
?4
@5
A6
l

%kernel
&bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

!kernel
"bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

#kernel
$bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
!0
"1
#2
$3
%4
&5
*
!0
"1
#2
$3
%4
&5
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
1
I0
J1
K2
L3
M4
N5
O6
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

'kernel
(bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

)kernel
*bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

'0
(1
)2
*3

'0
(1
)2
*3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
1
W0
X1
Y2
Z3
[4
\5
]6
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

+kernel
,bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

-kernel
.bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

+0
,1
-2
.3

+0
,1
-2
.3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
1
e0
f1
g2
h3
i4
j5
k6
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

/kernel
0bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

1kernel
2bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

/0
01
12
23

/0
01
12
23
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api

%0
&1

%0
&1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

!0
"1

!0
"1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

#0
$1

#0
$1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
?
;0
<1
=2
>3
?4
@5
A6
97
:8
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

'0
(1

'0
(1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

)0
*1

)0
*1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
?
I0
J1
K2
L3
M4
N5
O6
G7
H8
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

+0
,1

+0
,1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

-0
.1

-0
.1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
?
W0
X1
Y2
Z3
[4
\5
]6
U7
V8
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

/0
01

/0
01
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

10
21

10
21
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
?
e0
f1
g2
h3
i4
j5
k6
c7
d8
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
vt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_0/conv1D_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_0/conv1D_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_0/conv1D_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_0/conv1D_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-tcn/residual_block_0/matching_conv1D/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+tcn/residual_block_0/matching_conv1D/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_1/conv1D_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_1/conv1D_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_1/conv1D_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_1/conv1D_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_2/conv1D_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_2/conv1D_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_2/conv1D_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_2/conv1D_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_3/conv1D_0/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_3/conv1D_0/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_3/conv1D_1/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_3/conv1D_1/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_0/conv1D_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_0/conv1D_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_0/conv1D_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_0/conv1D_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-tcn/residual_block_0/matching_conv1D/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+tcn/residual_block_0/matching_conv1D/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_1/conv1D_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_1/conv1D_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&tcn/residual_block_1/conv1D_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$tcn/residual_block_1/conv1D_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_2/conv1D_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_2/conv1D_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_2/conv1D_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_2/conv1D_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_3/conv1D_0/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_3/conv1D_0/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&tcn/residual_block_3/conv1D_1/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$tcn/residual_block_3/conv1D_1/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_tcn_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_tcn_input$tcn/residual_block_0/conv1D_0/kernel"tcn/residual_block_0/conv1D_0/bias$tcn/residual_block_0/conv1D_1/kernel"tcn/residual_block_0/conv1D_1/bias+tcn/residual_block_0/matching_conv1D/kernel)tcn/residual_block_0/matching_conv1D/bias$tcn/residual_block_1/conv1D_0/kernel"tcn/residual_block_1/conv1D_0/bias$tcn/residual_block_1/conv1D_1/kernel"tcn/residual_block_1/conv1D_1/bias$tcn/residual_block_2/conv1D_0/kernel"tcn/residual_block_2/conv1D_0/bias$tcn/residual_block_2/conv1D_1/kernel"tcn/residual_block_2/conv1D_1/bias$tcn/residual_block_3/conv1D_0/kernel"tcn/residual_block_3/conv1D_0/bias$tcn/residual_block_3/conv1D_1/kernel"tcn/residual_block_3/conv1D_1/biasdense/kernel
dense/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1154731
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp8tcn/residual_block_0/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_0/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_0/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_0/conv1D_1/bias/Read/ReadVariableOp?tcn/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOp=tcn/residual_block_0/matching_conv1D/bias/Read/ReadVariableOp8tcn/residual_block_1/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_1/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_1/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_1/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_2/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_2/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_2/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_2/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_3/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_3/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_3/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_3/conv1D_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp:tcn/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOp8tcn/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOp:tcn/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOp8tcn/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpAtcn/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOp?tcn/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOp:tcn/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOp8tcn/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOp:tcn/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOp8tcn/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOp:tcn/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOp8tcn/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOp:tcn/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOp8tcn/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOp:tcn/residual_block_3/conv1D_0/kernel/m/Read/ReadVariableOp8tcn/residual_block_3/conv1D_0/bias/m/Read/ReadVariableOp:tcn/residual_block_3/conv1D_1/kernel/m/Read/ReadVariableOp8tcn/residual_block_3/conv1D_1/bias/m/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOp:tcn/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOp8tcn/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOp:tcn/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOp8tcn/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpAtcn/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOp?tcn/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOp:tcn/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOp8tcn/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOp:tcn/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOp8tcn/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOp:tcn/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOp8tcn/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOp:tcn/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOp8tcn/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOp:tcn/residual_block_3/conv1D_0/kernel/v/Read/ReadVariableOp8tcn/residual_block_3/conv1D_0/bias/v/Read/ReadVariableOp:tcn/residual_block_3/conv1D_1/kernel/v/Read/ReadVariableOp8tcn/residual_block_3/conv1D_1/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1156402
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate$tcn/residual_block_0/conv1D_0/kernel"tcn/residual_block_0/conv1D_0/bias$tcn/residual_block_0/conv1D_1/kernel"tcn/residual_block_0/conv1D_1/bias+tcn/residual_block_0/matching_conv1D/kernel)tcn/residual_block_0/matching_conv1D/bias$tcn/residual_block_1/conv1D_0/kernel"tcn/residual_block_1/conv1D_0/bias$tcn/residual_block_1/conv1D_1/kernel"tcn/residual_block_1/conv1D_1/bias$tcn/residual_block_2/conv1D_0/kernel"tcn/residual_block_2/conv1D_0/bias$tcn/residual_block_2/conv1D_1/kernel"tcn/residual_block_2/conv1D_1/bias$tcn/residual_block_3/conv1D_0/kernel"tcn/residual_block_3/conv1D_0/bias$tcn/residual_block_3/conv1D_1/kernel"tcn/residual_block_3/conv1D_1/biastotalcounttotal_1count_1total_2count_2dense/kernel/mdense/bias/m&tcn/residual_block_0/conv1D_0/kernel/m$tcn/residual_block_0/conv1D_0/bias/m&tcn/residual_block_0/conv1D_1/kernel/m$tcn/residual_block_0/conv1D_1/bias/m-tcn/residual_block_0/matching_conv1D/kernel/m+tcn/residual_block_0/matching_conv1D/bias/m&tcn/residual_block_1/conv1D_0/kernel/m$tcn/residual_block_1/conv1D_0/bias/m&tcn/residual_block_1/conv1D_1/kernel/m$tcn/residual_block_1/conv1D_1/bias/m&tcn/residual_block_2/conv1D_0/kernel/m$tcn/residual_block_2/conv1D_0/bias/m&tcn/residual_block_2/conv1D_1/kernel/m$tcn/residual_block_2/conv1D_1/bias/m&tcn/residual_block_3/conv1D_0/kernel/m$tcn/residual_block_3/conv1D_0/bias/m&tcn/residual_block_3/conv1D_1/kernel/m$tcn/residual_block_3/conv1D_1/bias/mdense/kernel/vdense/bias/v&tcn/residual_block_0/conv1D_0/kernel/v$tcn/residual_block_0/conv1D_0/bias/v&tcn/residual_block_0/conv1D_1/kernel/v$tcn/residual_block_0/conv1D_1/bias/v-tcn/residual_block_0/matching_conv1D/kernel/v+tcn/residual_block_0/matching_conv1D/bias/v&tcn/residual_block_1/conv1D_0/kernel/v$tcn/residual_block_1/conv1D_0/bias/v&tcn/residual_block_1/conv1D_1/kernel/v$tcn/residual_block_1/conv1D_1/bias/v&tcn/residual_block_2/conv1D_0/kernel/v$tcn/residual_block_2/conv1D_0/bias/v&tcn/residual_block_2/conv1D_1/kernel/v$tcn/residual_block_2/conv1D_1/bias/v&tcn/residual_block_3/conv1D_0/kernel/v$tcn/residual_block_3/conv1D_0/bias/v&tcn/residual_block_3/conv1D_1/kernel/v$tcn/residual_block_3/conv1D_1/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1156625??
?
?
%__inference_tcn_layer_call_fn_1155378

inputs
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????n*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1153930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153610

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1155980

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153520v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1154678
	tcn_input!
tcn_1154635:n
tcn_1154637:n!
tcn_1154639:nn
tcn_1154641:n!
tcn_1154643:n
tcn_1154645:n!
tcn_1154647:nn
tcn_1154649:n!
tcn_1154651:nn
tcn_1154653:n!
tcn_1154655:nn
tcn_1154657:n!
tcn_1154659:nn
tcn_1154661:n!
tcn_1154663:nn
tcn_1154665:n!
tcn_1154667:nn
tcn_1154669:n
dense_1154672:n

dense_1154674:

identity??dense/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCall	tcn_inputtcn_1154635tcn_1154637tcn_1154639tcn_1154641tcn_1154643tcn_1154645tcn_1154647tcn_1154649tcn_1154651tcn_1154653tcn_1154655tcn_1154657tcn_1154659tcn_1154661tcn_1154663tcn_1154665tcn_1154667tcn_1154669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????n*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1154366?
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_1154672dense_1154674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1153978u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153670

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153502

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1155998

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
à
?
@__inference_tcn_layer_call_and_return_conditional_losses_1154366

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:nb
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:nN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:n
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_0/SDropout_0/ShapeShape0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_0/SDropout_0/strided_sliceStridedSlice*residual_block_0/SDropout_0/Shape:output:08residual_block_0/SDropout_0/strided_slice/stack:output:0:residual_block_0/SDropout_0/strided_slice/stack_1:output:0:residual_block_0/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_0/SDropout_0/strided_slice_1StridedSlice*residual_block_0/SDropout_0/Shape:output:0:residual_block_0/SDropout_0/strided_slice_1/stack:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad0residual_block_0/Act_Conv1D_0/Relu:activations:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_0/SDropout_1/ShapeShape0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_0/SDropout_1/strided_sliceStridedSlice*residual_block_0/SDropout_1/Shape:output:08residual_block_0/SDropout_1/strided_slice/stack:output:0:residual_block_0/SDropout_1/strided_slice/stack_1:output:0:residual_block_0/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_0/SDropout_1/strided_slice_1StridedSlice*residual_block_0/SDropout_1/Shape:output:0:residual_block_0/SDropout_1/strided_slice_1/stack:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_0/Act_Conv_Blocks/ReluRelu0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_1/SDropout_0/ShapeShape0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_1/SDropout_0/strided_sliceStridedSlice*residual_block_1/SDropout_0/Shape:output:08residual_block_1/SDropout_0/strided_slice/stack:output:0:residual_block_1/SDropout_0/strided_slice/stack_1:output:0:residual_block_1/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_1/SDropout_0/strided_slice_1StridedSlice*residual_block_1/SDropout_0/Shape:output:0:residual_block_1/SDropout_0/strided_slice_1/stack:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad0residual_block_1/Act_Conv1D_0/Relu:activations:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_1/SDropout_1/ShapeShape0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_1/SDropout_1/strided_sliceStridedSlice*residual_block_1/SDropout_1/Shape:output:08residual_block_1/SDropout_1/strided_slice/stack:output:0:residual_block_1/SDropout_1/strided_slice/stack_1:output:0:residual_block_1/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_1/SDropout_1/strided_slice_1StridedSlice*residual_block_1/SDropout_1/Shape:output:0:residual_block_1/SDropout_1/strided_slice_1/stack:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_1/Act_Conv_Blocks/ReluRelu0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_2/SDropout_0/ShapeShape0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_2/SDropout_0/strided_sliceStridedSlice*residual_block_2/SDropout_0/Shape:output:08residual_block_2/SDropout_0/strided_slice/stack:output:0:residual_block_2/SDropout_0/strided_slice/stack_1:output:0:residual_block_2/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_2/SDropout_0/strided_slice_1StridedSlice*residual_block_2/SDropout_0/Shape:output:0:residual_block_2/SDropout_0/strided_slice_1/stack:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad0residual_block_2/Act_Conv1D_0/Relu:activations:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_2/SDropout_1/ShapeShape0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_2/SDropout_1/strided_sliceStridedSlice*residual_block_2/SDropout_1/Shape:output:08residual_block_2/SDropout_1/strided_slice/stack:output:0:residual_block_2/SDropout_1/strided_slice/stack_1:output:0:residual_block_2/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_2/SDropout_1/strided_slice_1StridedSlice*residual_block_2/SDropout_1/Shape:output:0:residual_block_2/SDropout_1/strided_slice_1/stack:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_2/Act_Conv_Blocks/ReluRelu0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_3/SDropout_0/ShapeShape0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_3/SDropout_0/strided_sliceStridedSlice*residual_block_3/SDropout_0/Shape:output:08residual_block_3/SDropout_0/strided_slice/stack:output:0:residual_block_3/SDropout_0/strided_slice/stack_1:output:0:residual_block_3/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_3/SDropout_0/strided_slice_1StridedSlice*residual_block_3/SDropout_0/Shape:output:0:residual_block_3/SDropout_0/strided_slice_1/stack:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad0residual_block_3/Act_Conv1D_0/Relu:activations:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_3/SDropout_1/ShapeShape0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_3/SDropout_1/strided_sliceStridedSlice*residual_block_3/SDropout_1/Shape:output:08residual_block_3/SDropout_1/strided_slice/stack:output:0:residual_block_3/SDropout_1/strided_slice/stack_1:output:0:residual_block_3/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_3/SDropout_1/strided_slice_1StridedSlice*residual_block_3/SDropout_1/Shape:output:0:residual_block_3/SDropout_1/strided_slice_1/stack:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_3/Act_Conv_Blocks/ReluRelu0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????nu
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????n?
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1154632
	tcn_input!
tcn_1154589:n
tcn_1154591:n!
tcn_1154593:nn
tcn_1154595:n!
tcn_1154597:n
tcn_1154599:n!
tcn_1154601:nn
tcn_1154603:n!
tcn_1154605:nn
tcn_1154607:n!
tcn_1154609:nn
tcn_1154611:n!
tcn_1154613:nn
tcn_1154615:n!
tcn_1154617:nn
tcn_1154619:n!
tcn_1154621:nn
tcn_1154623:n
dense_1154626:n

dense_1154628:

identity??dense/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCall	tcn_inputtcn_1154589tcn_1154591tcn_1154593tcn_1154595tcn_1154597tcn_1154599tcn_1154601tcn_1154603tcn_1154605tcn_1154607tcn_1154609tcn_1154611tcn_1154613tcn_1154615tcn_1154617tcn_1154619tcn_1154621tcn_1154623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????n*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1153930?
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_1154626dense_1154628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1153978u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156026

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_tcn_layer_call_and_return_conditional_losses_1153930

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:nb
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:nN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:n
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????nu
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????n?
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153592

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_1153978

inputs0
matmul_readvariableop_resource:n
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:n
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????n: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????n
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153532

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156125

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156153

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?$
 __inference__traced_save_1156402
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_0_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_0_conv1d_1_bias_read_readvariableopJ
Fsavev2_tcn_residual_block_0_matching_conv1d_kernel_read_readvariableopH
Dsavev2_tcn_residual_block_0_matching_conv1d_bias_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_1_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_1_conv1d_1_bias_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_2_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_2_conv1d_1_bias_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_3_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_3_conv1d_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_0_conv1d_0_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_0_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_0_conv1d_1_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_1_bias_m_read_readvariableopL
Hsavev2_tcn_residual_block_0_matching_conv1d_kernel_m_read_readvariableopJ
Fsavev2_tcn_residual_block_0_matching_conv1d_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_1_conv1d_0_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_0_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_1_conv1d_1_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_1_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_2_conv1d_0_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_0_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_2_conv1d_1_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_1_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_3_conv1d_0_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_0_bias_m_read_readvariableopE
Asavev2_tcn_residual_block_3_conv1d_1_kernel_m_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_1_bias_m_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_0_conv1d_0_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_0_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_0_conv1d_1_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_1_bias_v_read_readvariableopL
Hsavev2_tcn_residual_block_0_matching_conv1d_kernel_v_read_readvariableopJ
Fsavev2_tcn_residual_block_0_matching_conv1d_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_1_conv1d_0_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_0_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_1_conv1d_1_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_1_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_2_conv1d_0_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_0_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_2_conv1d_1_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_1_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_3_conv1d_0_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_0_bias_v_read_readvariableopE
Asavev2_tcn_residual_block_3_conv1d_1_kernel_v_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_1_bias_v_read_readvariableop
savev2_const

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
: ?!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?!
value?!B?!HB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop?savev2_tcn_residual_block_0_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_0_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_0_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_0_conv1d_1_bias_read_readvariableopFsavev2_tcn_residual_block_0_matching_conv1d_kernel_read_readvariableopDsavev2_tcn_residual_block_0_matching_conv1d_bias_read_readvariableop?savev2_tcn_residual_block_1_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_1_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_1_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_1_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_2_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_2_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_2_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_2_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_3_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_3_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_3_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_3_conv1d_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableopAsavev2_tcn_residual_block_0_conv1d_0_kernel_m_read_readvariableop?savev2_tcn_residual_block_0_conv1d_0_bias_m_read_readvariableopAsavev2_tcn_residual_block_0_conv1d_1_kernel_m_read_readvariableop?savev2_tcn_residual_block_0_conv1d_1_bias_m_read_readvariableopHsavev2_tcn_residual_block_0_matching_conv1d_kernel_m_read_readvariableopFsavev2_tcn_residual_block_0_matching_conv1d_bias_m_read_readvariableopAsavev2_tcn_residual_block_1_conv1d_0_kernel_m_read_readvariableop?savev2_tcn_residual_block_1_conv1d_0_bias_m_read_readvariableopAsavev2_tcn_residual_block_1_conv1d_1_kernel_m_read_readvariableop?savev2_tcn_residual_block_1_conv1d_1_bias_m_read_readvariableopAsavev2_tcn_residual_block_2_conv1d_0_kernel_m_read_readvariableop?savev2_tcn_residual_block_2_conv1d_0_bias_m_read_readvariableopAsavev2_tcn_residual_block_2_conv1d_1_kernel_m_read_readvariableop?savev2_tcn_residual_block_2_conv1d_1_bias_m_read_readvariableopAsavev2_tcn_residual_block_3_conv1d_0_kernel_m_read_readvariableop?savev2_tcn_residual_block_3_conv1d_0_bias_m_read_readvariableopAsavev2_tcn_residual_block_3_conv1d_1_kernel_m_read_readvariableop?savev2_tcn_residual_block_3_conv1d_1_bias_m_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableopAsavev2_tcn_residual_block_0_conv1d_0_kernel_v_read_readvariableop?savev2_tcn_residual_block_0_conv1d_0_bias_v_read_readvariableopAsavev2_tcn_residual_block_0_conv1d_1_kernel_v_read_readvariableop?savev2_tcn_residual_block_0_conv1d_1_bias_v_read_readvariableopHsavev2_tcn_residual_block_0_matching_conv1d_kernel_v_read_readvariableopFsavev2_tcn_residual_block_0_matching_conv1d_bias_v_read_readvariableopAsavev2_tcn_residual_block_1_conv1d_0_kernel_v_read_readvariableop?savev2_tcn_residual_block_1_conv1d_0_bias_v_read_readvariableopAsavev2_tcn_residual_block_1_conv1d_1_kernel_v_read_readvariableop?savev2_tcn_residual_block_1_conv1d_1_bias_v_read_readvariableopAsavev2_tcn_residual_block_2_conv1d_0_kernel_v_read_readvariableop?savev2_tcn_residual_block_2_conv1d_0_bias_v_read_readvariableopAsavev2_tcn_residual_block_2_conv1d_1_kernel_v_read_readvariableop?savev2_tcn_residual_block_2_conv1d_1_bias_v_read_readvariableopAsavev2_tcn_residual_block_3_conv1d_0_kernel_v_read_readvariableop?savev2_tcn_residual_block_3_conv1d_0_bias_v_read_readvariableopAsavev2_tcn_residual_block_3_conv1d_1_kernel_v_read_readvariableop?savev2_tcn_residual_block_3_conv1d_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :n
:
: : : : : :n:n:nn:n:n:n:nn:n:nn:n:nn:n:nn:n:nn:n:nn:n: : : : : : :n
:
:n:n:nn:n:n:n:nn:n:nn:n:nn:n:nn:n:nn:n:nn:n:n
:
:n:n:nn:n:n:n:nn:n:nn:n:nn:n:nn:n:nn:n:nn:n: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:n
: 

_output_shapes
:
:

_output_shapes
: :
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
: :($
"
_output_shapes
:n: 	

_output_shapes
:n:(
$
"
_output_shapes
:nn: 

_output_shapes
:n:($
"
_output_shapes
:n: 

_output_shapes
:n:($
"
_output_shapes
:nn: 

_output_shapes
:n:($
"
_output_shapes
:nn: 

_output_shapes
:n:($
"
_output_shapes
:nn: 

_output_shapes
:n:($
"
_output_shapes
:nn: 

_output_shapes
:n:($
"
_output_shapes
:nn: 

_output_shapes
:n:($
"
_output_shapes
:nn: 

_output_shapes
:n:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$  

_output_shapes

:n
: !

_output_shapes
:
:("$
"
_output_shapes
:n: #

_output_shapes
:n:($$
"
_output_shapes
:nn: %

_output_shapes
:n:(&$
"
_output_shapes
:n: '

_output_shapes
:n:(($
"
_output_shapes
:nn: )

_output_shapes
:n:(*$
"
_output_shapes
:nn: +

_output_shapes
:n:(,$
"
_output_shapes
:nn: -

_output_shapes
:n:(.$
"
_output_shapes
:nn: /

_output_shapes
:n:(0$
"
_output_shapes
:nn: 1

_output_shapes
:n:(2$
"
_output_shapes
:nn: 3

_output_shapes
:n:$4 

_output_shapes

:n
: 5

_output_shapes
:
:(6$
"
_output_shapes
:n: 7

_output_shapes
:n:(8$
"
_output_shapes
:nn: 9

_output_shapes
:n:(:$
"
_output_shapes
:n: ;

_output_shapes
:n:(<$
"
_output_shapes
:nn: =

_output_shapes
:n:(>$
"
_output_shapes
:nn: ?

_output_shapes
:n:(@$
"
_output_shapes
:nn: A

_output_shapes
:n:(B$
"
_output_shapes
:nn: C

_output_shapes
:n:(D$
"
_output_shapes
:nn: E

_output_shapes
:n:(F$
"
_output_shapes
:nn: G

_output_shapes
:n:H

_output_shapes
: 
?
H
,__inference_SDropout_0_layer_call_fn_1156120

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153670v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153652

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153640

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_1154776

inputs
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n

unknown_17:n


unknown_18:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1153985o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153682

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_tcn_layer_call_and_return_conditional_losses_1155639

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:nb
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:nN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:n
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????nu
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????n?
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1155952

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153490v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153622

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156013

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1153985

inputs!
tcn_1153931:n
tcn_1153933:n!
tcn_1153935:nn
tcn_1153937:n!
tcn_1153939:n
tcn_1153941:n!
tcn_1153943:nn
tcn_1153945:n!
tcn_1153947:nn
tcn_1153949:n!
tcn_1153951:nn
tcn_1153953:n!
tcn_1153955:nn
tcn_1153957:n!
tcn_1153959:nn
tcn_1153961:n!
tcn_1153963:nn
tcn_1153965:n
dense_1153979:n

dense_1153981:

identity??dense/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCallinputstcn_1153931tcn_1153933tcn_1153935tcn_1153937tcn_1153939tcn_1153941tcn_1153943tcn_1153945tcn_1153947tcn_1153949tcn_1153951tcn_1153953tcn_1153955tcn_1153957tcn_1153959tcn_1153961tcn_1153963tcn_1153965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????n*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1153930?
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_1153979dense_1153981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1153978u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1156148

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153700v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_1154821

inputs
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n

unknown_17:n


unknown_18:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1154498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1156003

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153532v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_1154586
	tcn_input
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n

unknown_17:n


unknown_18:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	tcn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1154498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
ݺ
?
G__inference_sequential_layer_call_and_return_conditional_losses_1155337

inputs_
Itcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nK
=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:nf
Ptcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:nR
Dtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:n_
Itcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:n_
Itcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:n_
Itcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:n6
$dense_matmul_readvariableop_resource:n
3
%dense_biasadd_readvariableop_resource:

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
*tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_0/PadPadinputs3tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????~
3tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_0/Pad:output:0<tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0w
5tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
$tcn/residual_block_0/conv1D_0/Conv1DConv2D8tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_0/conv1D_0/BiasAddBiasAdd5tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_0/Act_Conv1D_0/ReluRelu.tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_0/SDropout_0/ShapeShape4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_0/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_0/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_0/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_0/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_0/SDropout_0/Shape:output:0<tcn/residual_block_0/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_0/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_0/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_0/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_0/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_0/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_0/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_0/SDropout_0/Shape:output:0>tcn/residual_block_0/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_0/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_0/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
*tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_1/PadPad4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:03tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_1/Pad:output:0<tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_0/conv1D_1/Conv1DConv2D8tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_0/conv1D_1/BiasAddBiasAdd5tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_0/Act_Conv1D_1/ReluRelu.tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_0/SDropout_1/ShapeShape4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_0/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_0/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_0/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_0/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_0/SDropout_1/Shape:output:0<tcn/residual_block_0/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_0/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_0/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_0/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_0/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_0/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_0/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_0/SDropout_1/Shape:output:0>tcn/residual_block_0/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_0/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_0/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)tcn/residual_block_0/Act_Conv_Blocks/ReluRelu4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
:tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputsCtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpPtcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0~
<tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
8tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsOtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Etcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
+tcn/residual_block_0/matching_conv1D/Conv1DConv2D?tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Atcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
3tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze4tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpDtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
,tcn/residual_block_0/matching_conv1D/BiasAddBiasAdd<tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Ctcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_0/Add_Res/addAddV25tcn/residual_block_0/matching_conv1D/BiasAdd:output:07tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_0/Act_Res_Block/ReluRelu$tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_0/PadPad5tcn/residual_block_0/Act_Res_Block/Relu:activations:03tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_0/Pad:output:0Htcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
n~
3tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_1/conv1D_0/Conv1DConv2D8tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_1/conv1D_0/BiasAddBiasAdd<tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_1/Act_Conv1D_0/ReluRelu.tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_1/SDropout_0/ShapeShape4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_1/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_1/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_1/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_1/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_1/SDropout_0/Shape:output:0<tcn/residual_block_1/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_1/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_1/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_1/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_1/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_1/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_1/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_1/SDropout_0/Shape:output:0>tcn/residual_block_1/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_1/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_1/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
*tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_1/PadPad4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:03tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_1/Pad:output:0Htcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
n~
3tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_1/conv1D_1/Conv1DConv2D8tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_1/conv1D_1/BiasAddBiasAdd<tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_1/Act_Conv1D_1/ReluRelu.tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_1/SDropout_1/ShapeShape4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_1/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_1/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_1/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_1/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_1/SDropout_1/Shape:output:0<tcn/residual_block_1/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_1/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_1/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_1/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_1/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_1/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_1/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_1/SDropout_1/Shape:output:0>tcn/residual_block_1/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_1/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_1/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)tcn/residual_block_1/Act_Conv_Blocks/ReluRelu4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_1/Add_Res/addAddV25tcn/residual_block_0/Act_Res_Block/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_1/Act_Res_Block/ReluRelu$tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_0/PadPad5tcn/residual_block_1/Act_Res_Block/Relu:activations:03tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_0/Pad:output:0Htcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_2/conv1D_0/Conv1DConv2D8tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_2/conv1D_0/BiasAddBiasAdd<tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_2/Act_Conv1D_0/ReluRelu.tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_2/SDropout_0/ShapeShape4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_2/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_2/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_2/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_2/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_2/SDropout_0/Shape:output:0<tcn/residual_block_2/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_2/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_2/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_2/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_2/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_2/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_2/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_2/SDropout_0/Shape:output:0>tcn/residual_block_2/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_2/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_2/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
*tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_1/PadPad4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:03tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_1/Pad:output:0Htcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_2/conv1D_1/Conv1DConv2D8tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_2/conv1D_1/BiasAddBiasAdd<tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_2/Act_Conv1D_1/ReluRelu.tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_2/SDropout_1/ShapeShape4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_2/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_2/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_2/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_2/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_2/SDropout_1/Shape:output:0<tcn/residual_block_2/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_2/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_2/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_2/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_2/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_2/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_2/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_2/SDropout_1/Shape:output:0>tcn/residual_block_2/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_2/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_2/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)tcn/residual_block_2/Act_Conv_Blocks/ReluRelu4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_2/Add_Res/addAddV25tcn/residual_block_1/Act_Res_Block/Relu:activations:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_2/Act_Res_Block/ReluRelu$tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_0/PadPad5tcn/residual_block_2/Act_Res_Block/Relu:activations:03tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? n|
2tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Stcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_0/Pad:output:0Htcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_3/conv1D_0/Conv1DConv2D8tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_3/conv1D_0/BiasAddBiasAdd<tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_3/Act_Conv1D_0/ReluRelu.tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_3/SDropout_0/ShapeShape4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_3/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_3/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_3/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_3/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_3/SDropout_0/Shape:output:0<tcn/residual_block_3/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_3/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_3/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_3/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_3/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_3/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_3/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_3/SDropout_0/Shape:output:0>tcn/residual_block_3/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_3/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_3/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
*tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_1/PadPad4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:03tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? n|
2tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Stcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_1/Pad:output:0Htcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_3/conv1D_1/Conv1DConv2D8tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_3/conv1D_1/BiasAddBiasAdd<tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_3/Act_Conv1D_1/ReluRelu.tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
%tcn/residual_block_3/SDropout_1/ShapeShape4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_3/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_3/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_3/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-tcn/residual_block_3/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_3/SDropout_1/Shape:output:0<tcn/residual_block_3/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_3/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_3/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_3/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_3/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7tcn/residual_block_3/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/tcn/residual_block_3/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_3/SDropout_1/Shape:output:0>tcn/residual_block_3/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_3/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_3/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)tcn/residual_block_3/Act_Conv_Blocks/ReluRelu4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_3/Add_Res/addAddV25tcn/residual_block_2/Act_Res_Block/Relu:activations:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_3/Act_Res_Block/ReluRelu$tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
tcn/Add_Skip_Connections/addAddV27tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
tcn/Add_Skip_Connections/add_1AddV2 tcn/Add_Skip_Connections/add:z:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
tcn/Add_Skip_Connections/add_2AddV2"tcn/Add_Skip_Connections/add_1:z:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????ny
$tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    {
&tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            {
&tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
tcn/Slice_Output/strided_sliceStridedSlice"tcn/Add_Skip_Connections/add_2:z:0-tcn/Slice_Output/strided_slice/stack:output:0/tcn/Slice_Output/strided_slice/stack_1:output:0/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_mask?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:n
*
dtype0?
dense/MatMulMatMul'tcn/Slice_Output/strided_slice:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp5^tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<^tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpH^tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2l
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2z
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpGtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153520

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1155970

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156069

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1156143

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153682v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_1155942

inputs0
matmul_readvariableop_resource:n
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:n
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????n: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????n
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153562

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1156059

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153592v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153700

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156110

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_dense_layer_call_fn_1155932

inputs
unknown:n

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1153978o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????n: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????n
 
_user_specified_nameinputs
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153550

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1155957

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1154498

inputs!
tcn_1154455:n
tcn_1154457:n!
tcn_1154459:nn
tcn_1154461:n!
tcn_1154463:n
tcn_1154465:n!
tcn_1154467:nn
tcn_1154469:n!
tcn_1154471:nn
tcn_1154473:n!
tcn_1154475:nn
tcn_1154477:n!
tcn_1154479:nn
tcn_1154481:n!
tcn_1154483:nn
tcn_1154485:n!
tcn_1154487:nn
tcn_1154489:n
dense_1154492:n

dense_1154494:

identity??dense/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCallinputstcn_1154455tcn_1154457tcn_1154459tcn_1154461tcn_1154463tcn_1154465tcn_1154467tcn_1154469tcn_1154471tcn_1154473tcn_1154475tcn_1154477tcn_1154479tcn_1154481tcn_1154483tcn_1154485tcn_1154487tcn_1154489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????n*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1154366?
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_1154492dense_1154494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1153978u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_tcn_layer_call_fn_1155419

inputs
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????n*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1154366o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1156008

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153550v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_1153463
	tcn_inputj
Tsequential_tcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nV
Hsequential_tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:nq
[sequential_tcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:n]
Osequential_tcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:nj
Tsequential_tcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnV
Hsequential_tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:nA
/sequential_dense_matmul_readvariableop_resource:n
>
0sequential_dense_biasadd_readvariableop_resource:

identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp??sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?Fsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Rsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp??sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?Ksequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
5sequential/tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_0/conv1D_0/PadPad	tcn_input>sequential/tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:??????????
>sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims5sequential/tcn/residual_block_0/conv1D_0/Pad:output:0Gsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Ksequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0?
@sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
/sequential/tcn/residual_block_0/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_0/conv1D_0/BiasAddBiasAdd@sequential/tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0Gsequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_0/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_0/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_0/conv1D_1/PadPad<sequential/tcn/residual_block_0/SDropout_0/Identity:output:0>sequential/tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n?
>sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims5sequential/tcn/residual_block_0/conv1D_1/Pad:output:0Gsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
Ksequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_0/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_0/conv1D_1/BiasAddBiasAdd@sequential/tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0Gsequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_0/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_0/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
4sequential/tcn/residual_block_0/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
Esequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Asequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDims	tcn_inputNsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Rsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp[sequential_tcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0?
Gsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsZsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Psequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
6sequential/tcn/residual_block_0/matching_conv1D/Conv1DConv2DJsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Lsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
>sequential/tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze?sequential/tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Fsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpOsequential_tcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
7sequential/tcn/residual_block_0/matching_conv1D/BiasAddBiasAddGsequential/tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Nsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
+sequential/tcn/residual_block_0/Add_Res/addAddV2@sequential/tcn/residual_block_0/matching_conv1D/BiasAdd:output:0Bsequential/tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
2sequential/tcn/residual_block_0/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_1/conv1D_0/PadPad@sequential/tcn/residual_block_0/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n?
=sequential/tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
\sequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ysequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Vsequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jsequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_1/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
n?
>sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
Ksequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_1/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Jsequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_1/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_1/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_1/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_1/conv1D_1/PadPad<sequential/tcn/residual_block_1/SDropout_0/Identity:output:0>sequential/tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n?
=sequential/tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
\sequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ysequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Vsequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jsequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_1/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
n?
>sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
Ksequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_1/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Jsequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_1/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_1/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_1/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
4sequential/tcn/residual_block_1/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
+sequential/tcn/residual_block_1/Add_Res/addAddV2@sequential/tcn/residual_block_0/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
2sequential/tcn/residual_block_1/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_2/conv1D_0/PadPad@sequential/tcn/residual_block_1/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n?
=sequential/tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
\sequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ysequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Vsequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jsequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_2/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n?
>sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
Ksequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_2/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Jsequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_2/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_2/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_2/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_2/conv1D_1/PadPad<sequential/tcn/residual_block_2/SDropout_0/Identity:output:0>sequential/tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n?
=sequential/tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
\sequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ysequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Vsequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jsequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_2/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n?
>sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
Ksequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_2/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Jsequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_2/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_2/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_2/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
4sequential/tcn/residual_block_2/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
+sequential/tcn/residual_block_2/Add_Res/addAddV2@sequential/tcn/residual_block_1/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
2sequential/tcn/residual_block_2/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_3/conv1D_0/PadPad@sequential/tcn/residual_block_2/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? n?
=sequential/tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
\sequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
^sequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ysequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Vsequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jsequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_3/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n?
>sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
Ksequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_3/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Jsequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_3/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_3/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_3/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
5sequential/tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
,sequential/tcn/residual_block_3/conv1D_1/PadPad<sequential/tcn/residual_block_3/SDropout_0/Identity:output:0>sequential/tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? n?
=sequential/tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
\sequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
^sequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ysequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Vsequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jsequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_3/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n?
>sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
Ksequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0?
@sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
/sequential/tcn/residual_block_3/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
7sequential/tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
Jsequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
>sequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
0sequential/tcn/residual_block_3/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
1sequential/tcn/residual_block_3/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
3sequential/tcn/residual_block_3/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
4sequential/tcn/residual_block_3/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
+sequential/tcn/residual_block_3/Add_Res/addAddV2@sequential/tcn/residual_block_2/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
2sequential/tcn/residual_block_3/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
'sequential/tcn/Add_Skip_Connections/addAddV2Bsequential/tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0Bsequential/tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
)sequential/tcn/Add_Skip_Connections/add_1AddV2+sequential/tcn/Add_Skip_Connections/add:z:0Bsequential/tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
)sequential/tcn/Add_Skip_Connections/add_2AddV2-sequential/tcn/Add_Skip_Connections/add_1:z:0Bsequential/tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
/sequential/tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    ?
1sequential/tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            ?
1sequential/tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
)sequential/tcn/Slice_Output/strided_sliceStridedSlice-sequential/tcn/Add_Skip_Connections/add_2:z:08sequential/tcn/Slice_Output/strided_slice/stack:output:0:sequential/tcn/Slice_Output/strided_slice/stack_1:output:0:sequential/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_mask?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:n
*
dtype0?
sequential/dense/MatMulMatMul2sequential/tcn/Slice_Output/strided_slice:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
IdentityIdentity!sequential/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp@^sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpG^sequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpS^sequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2?
?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2?
Fsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpFsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Rsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpRsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2?
Ksequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156138

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1156087

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153622v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156097

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1156036

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153580v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156166

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153472

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
à
?
@__inference_tcn_layer_call_and_return_conditional_losses_1155923

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:nb
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:nN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:n[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:n
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_0/SDropout_0/ShapeShape0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_0/SDropout_0/strided_sliceStridedSlice*residual_block_0/SDropout_0/Shape:output:08residual_block_0/SDropout_0/strided_slice/stack:output:0:residual_block_0/SDropout_0/strided_slice/stack_1:output:0:residual_block_0/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_0/SDropout_0/strided_slice_1StridedSlice*residual_block_0/SDropout_0/Shape:output:0:residual_block_0/SDropout_0/strided_slice_1/stack:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad0residual_block_0/Act_Conv1D_0/Relu:activations:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_0/SDropout_1/ShapeShape0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_0/SDropout_1/strided_sliceStridedSlice*residual_block_0/SDropout_1/Shape:output:08residual_block_0/SDropout_1/strided_slice/stack:output:0:residual_block_0/SDropout_1/strided_slice/stack_1:output:0:residual_block_0/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_0/SDropout_1/strided_slice_1StridedSlice*residual_block_0/SDropout_1/Shape:output:0:residual_block_0/SDropout_1/strided_slice_1/stack:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_0/Act_Conv_Blocks/ReluRelu0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_1/SDropout_0/ShapeShape0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_1/SDropout_0/strided_sliceStridedSlice*residual_block_1/SDropout_0/Shape:output:08residual_block_1/SDropout_0/strided_slice/stack:output:0:residual_block_1/SDropout_0/strided_slice/stack_1:output:0:residual_block_1/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_1/SDropout_0/strided_slice_1StridedSlice*residual_block_1/SDropout_0/Shape:output:0:residual_block_1/SDropout_0/strided_slice_1/stack:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad0residual_block_1/Act_Conv1D_0/Relu:activations:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
nz
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_1/SDropout_1/ShapeShape0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_1/SDropout_1/strided_sliceStridedSlice*residual_block_1/SDropout_1/Shape:output:08residual_block_1/SDropout_1/strided_slice/stack:output:0:residual_block_1/SDropout_1/strided_slice/stack_1:output:0:residual_block_1/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_1/SDropout_1/strided_slice_1StridedSlice*residual_block_1/SDropout_1/Shape:output:0:residual_block_1/SDropout_1/strided_slice_1/stack:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_1/Act_Conv_Blocks/ReluRelu0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_2/SDropout_0/ShapeShape0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_2/SDropout_0/strided_sliceStridedSlice*residual_block_2/SDropout_0/Shape:output:08residual_block_2/SDropout_0/strided_slice/stack:output:0:residual_block_2/SDropout_0/strided_slice/stack_1:output:0:residual_block_2/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_2/SDropout_0/strided_slice_1StridedSlice*residual_block_2/SDropout_0/Shape:output:0:residual_block_2/SDropout_0/strided_slice_1/stack:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad0residual_block_2/Act_Conv1D_0/Relu:activations:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????nx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_2/SDropout_1/ShapeShape0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_2/SDropout_1/strided_sliceStridedSlice*residual_block_2/SDropout_1/Shape:output:08residual_block_2/SDropout_1/strided_slice/stack:output:0:residual_block_2/SDropout_1/strided_slice/stack_1:output:0:residual_block_2/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_2/SDropout_1/strided_slice_1StridedSlice*residual_block_2/SDropout_1/Shape:output:0:residual_block_2/SDropout_1/strided_slice_1/stack:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_2/Act_Conv_Blocks/ReluRelu0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_3/SDropout_0/ShapeShape0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_3/SDropout_0/strided_sliceStridedSlice*residual_block_3/SDropout_0/Shape:output:08residual_block_3/SDropout_0/strided_slice/stack:output:0:residual_block_3/SDropout_0/strided_slice/stack_1:output:0:residual_block_3/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_3/SDropout_0/strided_slice_1StridedSlice*residual_block_3/SDropout_0/Shape:output:0:residual_block_3/SDropout_0/strided_slice_1/stack:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad0residual_block_3/Act_Conv1D_0/Relu:activations:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? nx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????nz
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
!residual_block_3/SDropout_1/ShapeShape0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)residual_block_3/SDropout_1/strided_sliceStridedSlice*residual_block_3/SDropout_1/Shape:output:08residual_block_3/SDropout_1/strided_slice/stack:output:0:residual_block_3/SDropout_1/strided_slice/stack_1:output:0:residual_block_3/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+residual_block_3/SDropout_1/strided_slice_1StridedSlice*residual_block_3/SDropout_1/Shape:output:0:residual_block_3/SDropout_1/strided_slice_1/stack:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%residual_block_3/Act_Conv_Blocks/ReluRelu0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????nu
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????n?
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1156031

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153562v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153580

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1156064

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153610v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156054

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153490

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1155985

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156082

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156041

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1156115

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153652v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?2
#__inference__traced_restore_1156625
file_prefix/
assignvariableop_dense_kernel:n
+
assignvariableop_1_dense_bias:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: M
7assignvariableop_7_tcn_residual_block_0_conv1d_0_kernel:nC
5assignvariableop_8_tcn_residual_block_0_conv1d_0_bias:nM
7assignvariableop_9_tcn_residual_block_0_conv1d_1_kernel:nnD
6assignvariableop_10_tcn_residual_block_0_conv1d_1_bias:nU
?assignvariableop_11_tcn_residual_block_0_matching_conv1d_kernel:nK
=assignvariableop_12_tcn_residual_block_0_matching_conv1d_bias:nN
8assignvariableop_13_tcn_residual_block_1_conv1d_0_kernel:nnD
6assignvariableop_14_tcn_residual_block_1_conv1d_0_bias:nN
8assignvariableop_15_tcn_residual_block_1_conv1d_1_kernel:nnD
6assignvariableop_16_tcn_residual_block_1_conv1d_1_bias:nN
8assignvariableop_17_tcn_residual_block_2_conv1d_0_kernel:nnD
6assignvariableop_18_tcn_residual_block_2_conv1d_0_bias:nN
8assignvariableop_19_tcn_residual_block_2_conv1d_1_kernel:nnD
6assignvariableop_20_tcn_residual_block_2_conv1d_1_bias:nN
8assignvariableop_21_tcn_residual_block_3_conv1d_0_kernel:nnD
6assignvariableop_22_tcn_residual_block_3_conv1d_0_bias:nN
8assignvariableop_23_tcn_residual_block_3_conv1d_1_kernel:nnD
6assignvariableop_24_tcn_residual_block_3_conv1d_1_bias:n#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: %
assignvariableop_29_total_2: %
assignvariableop_30_count_2: 4
"assignvariableop_31_dense_kernel_m:n
.
 assignvariableop_32_dense_bias_m:
P
:assignvariableop_33_tcn_residual_block_0_conv1d_0_kernel_m:nF
8assignvariableop_34_tcn_residual_block_0_conv1d_0_bias_m:nP
:assignvariableop_35_tcn_residual_block_0_conv1d_1_kernel_m:nnF
8assignvariableop_36_tcn_residual_block_0_conv1d_1_bias_m:nW
Aassignvariableop_37_tcn_residual_block_0_matching_conv1d_kernel_m:nM
?assignvariableop_38_tcn_residual_block_0_matching_conv1d_bias_m:nP
:assignvariableop_39_tcn_residual_block_1_conv1d_0_kernel_m:nnF
8assignvariableop_40_tcn_residual_block_1_conv1d_0_bias_m:nP
:assignvariableop_41_tcn_residual_block_1_conv1d_1_kernel_m:nnF
8assignvariableop_42_tcn_residual_block_1_conv1d_1_bias_m:nP
:assignvariableop_43_tcn_residual_block_2_conv1d_0_kernel_m:nnF
8assignvariableop_44_tcn_residual_block_2_conv1d_0_bias_m:nP
:assignvariableop_45_tcn_residual_block_2_conv1d_1_kernel_m:nnF
8assignvariableop_46_tcn_residual_block_2_conv1d_1_bias_m:nP
:assignvariableop_47_tcn_residual_block_3_conv1d_0_kernel_m:nnF
8assignvariableop_48_tcn_residual_block_3_conv1d_0_bias_m:nP
:assignvariableop_49_tcn_residual_block_3_conv1d_1_kernel_m:nnF
8assignvariableop_50_tcn_residual_block_3_conv1d_1_bias_m:n4
"assignvariableop_51_dense_kernel_v:n
.
 assignvariableop_52_dense_bias_v:
P
:assignvariableop_53_tcn_residual_block_0_conv1d_0_kernel_v:nF
8assignvariableop_54_tcn_residual_block_0_conv1d_0_bias_v:nP
:assignvariableop_55_tcn_residual_block_0_conv1d_1_kernel_v:nnF
8assignvariableop_56_tcn_residual_block_0_conv1d_1_bias_v:nW
Aassignvariableop_57_tcn_residual_block_0_matching_conv1d_kernel_v:nM
?assignvariableop_58_tcn_residual_block_0_matching_conv1d_bias_v:nP
:assignvariableop_59_tcn_residual_block_1_conv1d_0_kernel_v:nnF
8assignvariableop_60_tcn_residual_block_1_conv1d_0_bias_v:nP
:assignvariableop_61_tcn_residual_block_1_conv1d_1_kernel_v:nnF
8assignvariableop_62_tcn_residual_block_1_conv1d_1_bias_v:nP
:assignvariableop_63_tcn_residual_block_2_conv1d_0_kernel_v:nnF
8assignvariableop_64_tcn_residual_block_2_conv1d_0_bias_v:nP
:assignvariableop_65_tcn_residual_block_2_conv1d_1_kernel_v:nnF
8assignvariableop_66_tcn_residual_block_2_conv1d_1_bias_v:nP
:assignvariableop_67_tcn_residual_block_3_conv1d_0_kernel_v:nnF
8assignvariableop_68_tcn_residual_block_3_conv1d_0_bias_v:nP
:assignvariableop_69_tcn_residual_block_3_conv1d_1_kernel_v:nnF
8assignvariableop_70_tcn_residual_block_3_conv1d_1_bias_v:n
identity_72??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_8?AssignVariableOp_9?!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?!
value?!B?!HB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	[
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
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp7assignvariableop_7_tcn_residual_block_0_conv1d_0_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_tcn_residual_block_0_conv1d_0_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_tcn_residual_block_0_conv1d_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_tcn_residual_block_0_conv1d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp?assignvariableop_11_tcn_residual_block_0_matching_conv1d_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp=assignvariableop_12_tcn_residual_block_0_matching_conv1d_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp8assignvariableop_13_tcn_residual_block_1_conv1d_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_tcn_residual_block_1_conv1d_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp8assignvariableop_15_tcn_residual_block_1_conv1d_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_tcn_residual_block_1_conv1d_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp8assignvariableop_17_tcn_residual_block_2_conv1d_0_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_tcn_residual_block_2_conv1d_0_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp8assignvariableop_19_tcn_residual_block_2_conv1d_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_tcn_residual_block_2_conv1d_1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp8assignvariableop_21_tcn_residual_block_3_conv1d_0_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_tcn_residual_block_3_conv1d_0_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp8assignvariableop_23_tcn_residual_block_3_conv1d_1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_tcn_residual_block_3_conv1d_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp:assignvariableop_33_tcn_residual_block_0_conv1d_0_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp8assignvariableop_34_tcn_residual_block_0_conv1d_0_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_tcn_residual_block_0_conv1d_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp8assignvariableop_36_tcn_residual_block_0_conv1d_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpAassignvariableop_37_tcn_residual_block_0_matching_conv1d_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp?assignvariableop_38_tcn_residual_block_0_matching_conv1d_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp:assignvariableop_39_tcn_residual_block_1_conv1d_0_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp8assignvariableop_40_tcn_residual_block_1_conv1d_0_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp:assignvariableop_41_tcn_residual_block_1_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp8assignvariableop_42_tcn_residual_block_1_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp:assignvariableop_43_tcn_residual_block_2_conv1d_0_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp8assignvariableop_44_tcn_residual_block_2_conv1d_0_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp:assignvariableop_45_tcn_residual_block_2_conv1d_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp8assignvariableop_46_tcn_residual_block_2_conv1d_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp:assignvariableop_47_tcn_residual_block_3_conv1d_0_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp8assignvariableop_48_tcn_residual_block_3_conv1d_0_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp:assignvariableop_49_tcn_residual_block_3_conv1d_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp8assignvariableop_50_tcn_residual_block_3_conv1d_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp"assignvariableop_51_dense_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp assignvariableop_52_dense_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp:assignvariableop_53_tcn_residual_block_0_conv1d_0_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp8assignvariableop_54_tcn_residual_block_0_conv1d_0_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp:assignvariableop_55_tcn_residual_block_0_conv1d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp8assignvariableop_56_tcn_residual_block_0_conv1d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpAassignvariableop_57_tcn_residual_block_0_matching_conv1d_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp?assignvariableop_58_tcn_residual_block_0_matching_conv1d_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp:assignvariableop_59_tcn_residual_block_1_conv1d_0_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp8assignvariableop_60_tcn_residual_block_1_conv1d_0_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp:assignvariableop_61_tcn_residual_block_1_conv1d_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp8assignvariableop_62_tcn_residual_block_1_conv1d_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp:assignvariableop_63_tcn_residual_block_2_conv1d_0_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp8assignvariableop_64_tcn_residual_block_2_conv1d_0_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp:assignvariableop_65_tcn_residual_block_2_conv1d_1_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp8assignvariableop_66_tcn_residual_block_2_conv1d_1_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp:assignvariableop_67_tcn_residual_block_3_conv1d_0_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp8assignvariableop_68_tcn_residual_block_3_conv1d_0_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp:assignvariableop_69_tcn_residual_block_3_conv1d_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp8assignvariableop_70_tcn_residual_block_3_conv1d_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_72IdentityIdentity_71:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_72Identity_72:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
H
,__inference_SDropout_0_layer_call_fn_1155947

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1153472v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1154731
	tcn_input
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n

unknown_17:n


unknown_18:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	tcn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_1153463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
H
,__inference_SDropout_1_layer_call_fn_1156092

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153640v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_layer_call_and_return_conditional_losses_1155047

inputs_
Itcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nK
=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:nf
Ptcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:nR
Dtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:n_
Itcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:n_
Itcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:n_
Itcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:n_
Itcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:nnK
=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:n6
$dense_matmul_readvariableop_resource:n
3
%dense_biasadd_readvariableop_resource:

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
*tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_0/PadPadinputs3tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????~
3tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_0/Pad:output:0<tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0w
5tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
$tcn/residual_block_0/conv1D_0/Conv1DConv2D8tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_0/conv1D_0/BiasAddBiasAdd5tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_0/Act_Conv1D_0/ReluRelu.tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_0/SDropout_0/IdentityIdentity4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_1/PadPad1tcn/residual_block_0/SDropout_0/Identity:output:03tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_1/Pad:output:0<tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_0/conv1D_1/Conv1DConv2D8tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_0/conv1D_1/BiasAddBiasAdd5tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_0/Act_Conv1D_1/ReluRelu.tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_0/SDropout_1/IdentityIdentity4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
)tcn/residual_block_0/Act_Conv_Blocks/ReluRelu1tcn/residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
:tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputsCtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpPtcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:n*
dtype0~
<tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
8tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsOtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Etcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n?
+tcn/residual_block_0/matching_conv1D/Conv1DConv2D?tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Atcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingSAME*
strides
?
3tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze4tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpDtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
,tcn/residual_block_0/matching_conv1D/BiasAddBiasAdd<tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Ctcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_0/Add_Res/addAddV25tcn/residual_block_0/matching_conv1D/BiasAdd:output:07tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_0/Act_Res_Block/ReluRelu$tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_0/PadPad5tcn/residual_block_0/Act_Res_Block/Relu:activations:03tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_0/Pad:output:0Htcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
n~
3tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_1/conv1D_0/Conv1DConv2D8tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_1/conv1D_0/BiasAddBiasAdd<tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_1/Act_Conv1D_0/ReluRelu.tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_1/SDropout_0/IdentityIdentity4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_1/PadPad1tcn/residual_block_1/SDropout_0/Identity:output:03tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_1/Pad:output:0Htcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????
n~
3tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
n?
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_1/conv1D_1/Conv1DConv2D8tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_1/conv1D_1/BiasAddBiasAdd<tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_1/Act_Conv1D_1/ReluRelu.tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_1/SDropout_1/IdentityIdentity4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
)tcn/residual_block_1/Act_Conv_Blocks/ReluRelu1tcn/residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_1/Add_Res/addAddV25tcn/residual_block_0/Act_Res_Block/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_1/Act_Res_Block/ReluRelu$tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_0/PadPad5tcn/residual_block_1/Act_Res_Block/Relu:activations:03tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_0/Pad:output:0Htcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_2/conv1D_0/Conv1DConv2D8tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_2/conv1D_0/BiasAddBiasAdd<tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_2/Act_Conv1D_0/ReluRelu.tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_2/SDropout_0/IdentityIdentity4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_1/PadPad1tcn/residual_block_2/SDropout_0/Identity:output:03tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????n|
2tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Stcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_1/Pad:output:0Htcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_2/conv1D_1/Conv1DConv2D8tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_2/conv1D_1/BiasAddBiasAdd<tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_2/Act_Conv1D_1/ReluRelu.tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_2/SDropout_1/IdentityIdentity4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
)tcn/residual_block_2/Act_Conv_Blocks/ReluRelu1tcn/residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_2/Add_Res/addAddV25tcn/residual_block_1/Act_Res_Block/Relu:activations:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_2/Act_Res_Block/ReluRelu$tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_0/PadPad5tcn/residual_block_2/Act_Res_Block/Relu:activations:03tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? n|
2tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Stcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_0/Pad:output:0Htcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_3/conv1D_0/Conv1DConv2D8tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_3/conv1D_0/BiasAddBiasAdd<tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_3/Act_Conv1D_0/ReluRelu.tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_3/SDropout_0/IdentityIdentity4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
*tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_1/PadPad1tcn/residual_block_3/SDropout_0/Identity:output:03tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:????????? n|
2tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
Stcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ntcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Ktcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
?tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_1/Pad:output:0Htcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????n~
3tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????n?
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:nn*
dtype0w
5tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:nn?
$tcn/residual_block_3/conv1D_1/Conv1DConv2D8tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????n*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????n*
squeeze_dims

??????????
?tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
3tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????n?
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:n*
dtype0?
%tcn/residual_block_3/conv1D_1/BiasAddBiasAdd<tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????n?
&tcn/residual_block_3/Act_Conv1D_1/ReluRelu.tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????n?
(tcn/residual_block_3/SDropout_1/IdentityIdentity4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
)tcn/residual_block_3/Act_Conv_Blocks/ReluRelu1tcn/residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????n?
 tcn/residual_block_3/Add_Res/addAddV25tcn/residual_block_2/Act_Res_Block/Relu:activations:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
'tcn/residual_block_3/Act_Res_Block/ReluRelu$tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????n?
tcn/Add_Skip_Connections/addAddV27tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
tcn/Add_Skip_Connections/add_1AddV2 tcn/Add_Skip_Connections/add:z:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????n?
tcn/Add_Skip_Connections/add_2AddV2"tcn/Add_Skip_Connections/add_1:z:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????ny
$tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    {
&tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            {
&tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
tcn/Slice_Output/strided_sliceStridedSlice"tcn/Add_Skip_Connections/add_2:z:0-tcn/Slice_Output/strided_slice/stack:output:0/tcn/Slice_Output/strided_slice/stack_1:output:0/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????n*

begin_mask*
end_mask*
shrink_axis_mask?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:n
*
dtype0?
dense/MatMulMatMul'tcn/Slice_Output/strided_slice:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp5^tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<^tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpH^tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2l
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2z
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpGtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2?
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2?
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_1154028
	tcn_input
unknown:n
	unknown_0:n
	unknown_1:nn
	unknown_2:n
	unknown_3:n
	unknown_4:n
	unknown_5:nn
	unknown_6:n
	unknown_7:nn
	unknown_8:n
	unknown_9:nn

unknown_10:n 

unknown_11:nn

unknown_12:n 

unknown_13:nn

unknown_14:n 

unknown_15:nn

unknown_16:n

unknown_17:n


unknown_18:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	tcn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1153985o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
H
,__inference_SDropout_1_layer_call_fn_1155975

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1153502v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
	tcn_input6
serving_default_tcn_input:0?????????9
dense0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?
		dilations

skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
residual_block_3
slicer_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

beta_2
	decay
 learning_ratem?m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?v?v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?"
	optimizer
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
18
19"
trackable_list_wrapper
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?

8layers
9shape_match_conv
:final_activation
;conv1D_0
<Act_Conv1D_0
=
SDropout_0
>conv1D_1
?Act_Conv1D_1
@
SDropout_1
AAct_Conv_Blocks
9matching_conv1D
:Act_Res_Block
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Flayers
Gshape_match_conv
Hfinal_activation
Iconv1D_0
JAct_Conv1D_0
K
SDropout_0
Lconv1D_1
MAct_Conv1D_1
N
SDropout_1
OAct_Conv_Blocks
Gmatching_identity
HAct_Res_Block
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Tlayers
Ushape_match_conv
Vfinal_activation
Wconv1D_0
XAct_Conv1D_0
Y
SDropout_0
Zconv1D_1
[Act_Conv1D_1
\
SDropout_1
]Act_Conv_Blocks
Umatching_identity
VAct_Res_Block
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

blayers
cshape_match_conv
dfinal_activation
econv1D_0
fAct_Conv1D_0
g
SDropout_0
hconv1D_1
iAct_Conv1D_1
j
SDropout_1
kAct_Conv_Blocks
cmatching_identity
dAct_Res_Block
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217"
trackable_list_wrapper
?
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:n
2dense/kernel
:
2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
::8n2$tcn/residual_block_0/conv1D_0/kernel
0:.n2"tcn/residual_block_0/conv1D_0/bias
::8nn2$tcn/residual_block_0/conv1D_1/kernel
0:.n2"tcn/residual_block_0/conv1D_1/bias
A:?n2+tcn/residual_block_0/matching_conv1D/kernel
7:5n2)tcn/residual_block_0/matching_conv1D/bias
::8nn2$tcn/residual_block_1/conv1D_0/kernel
0:.n2"tcn/residual_block_1/conv1D_0/bias
::8nn2$tcn/residual_block_1/conv1D_1/kernel
0:.n2"tcn/residual_block_1/conv1D_1/bias
::8nn2$tcn/residual_block_2/conv1D_0/kernel
0:.n2"tcn/residual_block_2/conv1D_0/bias
::8nn2$tcn/residual_block_2/conv1D_1/kernel
0:.n2"tcn/residual_block_2/conv1D_1/bias
::8nn2$tcn/residual_block_3/conv1D_0/kernel
0:.n2"tcn/residual_block_3/conv1D_0/bias
::8nn2$tcn/residual_block_3/conv1D_1/kernel
0:.n2"tcn/residual_block_3/conv1D_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
6
~0
1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
;0
<1
=2
>3
?4
@5
A6"
trackable_list_wrapper
?

%kernel
&bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

!kernel
"bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

#kernel
$bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
Q
I0
J1
K2
L3
M4
N5
O6"
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
'0
(1
)2
*3"
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
Q
W0
X1
Y2
Z3
[4
\5
]6"
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

+kernel
,bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

-kernel
.bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
+0
,1
-2
.3"
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
Q
e0
f1
g2
h3
i4
j5
k6"
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
/0
01
12
23"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_
;0
<1
=2
>3
?4
@5
A6
97
:8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_
I0
J1
K2
L3
M4
N5
O6
G7
H8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_
W0
X1
Y2
Z3
[4
\5
]6
U7
V8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_
e0
f1
g2
h3
i4
j5
k6
c7
d8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
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
:n
2dense/kernel/m
:
2dense/bias/m
::8n2&tcn/residual_block_0/conv1D_0/kernel/m
0:.n2$tcn/residual_block_0/conv1D_0/bias/m
::8nn2&tcn/residual_block_0/conv1D_1/kernel/m
0:.n2$tcn/residual_block_0/conv1D_1/bias/m
A:?n2-tcn/residual_block_0/matching_conv1D/kernel/m
7:5n2+tcn/residual_block_0/matching_conv1D/bias/m
::8nn2&tcn/residual_block_1/conv1D_0/kernel/m
0:.n2$tcn/residual_block_1/conv1D_0/bias/m
::8nn2&tcn/residual_block_1/conv1D_1/kernel/m
0:.n2$tcn/residual_block_1/conv1D_1/bias/m
::8nn2&tcn/residual_block_2/conv1D_0/kernel/m
0:.n2$tcn/residual_block_2/conv1D_0/bias/m
::8nn2&tcn/residual_block_2/conv1D_1/kernel/m
0:.n2$tcn/residual_block_2/conv1D_1/bias/m
::8nn2&tcn/residual_block_3/conv1D_0/kernel/m
0:.n2$tcn/residual_block_3/conv1D_0/bias/m
::8nn2&tcn/residual_block_3/conv1D_1/kernel/m
0:.n2$tcn/residual_block_3/conv1D_1/bias/m
:n
2dense/kernel/v
:
2dense/bias/v
::8n2&tcn/residual_block_0/conv1D_0/kernel/v
0:.n2$tcn/residual_block_0/conv1D_0/bias/v
::8nn2&tcn/residual_block_0/conv1D_1/kernel/v
0:.n2$tcn/residual_block_0/conv1D_1/bias/v
A:?n2-tcn/residual_block_0/matching_conv1D/kernel/v
7:5n2+tcn/residual_block_0/matching_conv1D/bias/v
::8nn2&tcn/residual_block_1/conv1D_0/kernel/v
0:.n2$tcn/residual_block_1/conv1D_0/bias/v
::8nn2&tcn/residual_block_1/conv1D_1/kernel/v
0:.n2$tcn/residual_block_1/conv1D_1/bias/v
::8nn2&tcn/residual_block_2/conv1D_0/kernel/v
0:.n2$tcn/residual_block_2/conv1D_0/bias/v
::8nn2&tcn/residual_block_2/conv1D_1/kernel/v
0:.n2$tcn/residual_block_2/conv1D_1/bias/v
::8nn2&tcn/residual_block_3/conv1D_0/kernel/v
0:.n2$tcn/residual_block_3/conv1D_0/bias/v
::8nn2&tcn/residual_block_3/conv1D_1/kernel/v
0:.n2$tcn/residual_block_3/conv1D_1/bias/v
?2?
,__inference_sequential_layer_call_fn_1154028
,__inference_sequential_layer_call_fn_1154776
,__inference_sequential_layer_call_fn_1154821
,__inference_sequential_layer_call_fn_1154586?
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
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_1155047
G__inference_sequential_layer_call_and_return_conditional_losses_1155337
G__inference_sequential_layer_call_and_return_conditional_losses_1154632
G__inference_sequential_layer_call_and_return_conditional_losses_1154678?
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
?B?
"__inference__wrapped_model_1153463	tcn_input"?
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
?2?
%__inference_tcn_layer_call_fn_1155378
%__inference_tcn_layer_call_fn_1155419?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_tcn_layer_call_and_return_conditional_losses_1155639
@__inference_tcn_layer_call_and_return_conditional_losses_1155923?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_1155932?
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
?2?
B__inference_dense_layer_call_and_return_conditional_losses_1155942?
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
%__inference_signature_wrapper_1154731	tcn_input"?
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
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_0_layer_call_fn_1155947
,__inference_SDropout_0_layer_call_fn_1155952?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1155957
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1155970?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_1_layer_call_fn_1155975
,__inference_SDropout_1_layer_call_fn_1155980?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1155985
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1155998?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_0_layer_call_fn_1156003
,__inference_SDropout_0_layer_call_fn_1156008?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156013
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156026?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_1_layer_call_fn_1156031
,__inference_SDropout_1_layer_call_fn_1156036?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156041
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156054?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_0_layer_call_fn_1156059
,__inference_SDropout_0_layer_call_fn_1156064?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156069
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156082?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_1_layer_call_fn_1156087
,__inference_SDropout_1_layer_call_fn_1156092?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156097
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156110?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_0_layer_call_fn_1156115
,__inference_SDropout_0_layer_call_fn_1156120?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156125
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156138?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2?
,__inference_SDropout_1_layer_call_fn_1156143
,__inference_SDropout_1_layer_call_fn_1156148?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156153
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156166?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
 ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1155957?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1155970?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156013?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156026?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156069?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156082?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156125?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1156138?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
,__inference_SDropout_0_layer_call_fn_1155947{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1155952{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1156003{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1156008{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1156059{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1156064{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1156115{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1156120{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1155985?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1155998?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156041?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156054?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156097?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156110?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156153?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1156166?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
,__inference_SDropout_1_layer_call_fn_1155975{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1155980{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1156031{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1156036{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1156087{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1156092{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1156143{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1156148{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
"__inference__wrapped_model_1153463}!"#$%&'()*+,-./0126?3
,?)
'?$
	tcn_input?????????
? "-?*
(
dense?
dense?????????
?
B__inference_dense_layer_call_and_return_conditional_losses_1155942\/?,
%?"
 ?
inputs?????????n
? "%?"
?
0?????????

? z
'__inference_dense_layer_call_fn_1155932O/?,
%?"
 ?
inputs?????????n
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_1154632}!"#$%&'()*+,-./012>?;
4?1
'?$
	tcn_input?????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_sequential_layer_call_and_return_conditional_losses_1154678}!"#$%&'()*+,-./012>?;
4?1
'?$
	tcn_input?????????
p

 
? "%?"
?
0?????????

? ?
G__inference_sequential_layer_call_and_return_conditional_losses_1155047z!"#$%&'()*+,-./012;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_sequential_layer_call_and_return_conditional_losses_1155337z!"#$%&'()*+,-./012;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????

? ?
,__inference_sequential_layer_call_fn_1154028p!"#$%&'()*+,-./012>?;
4?1
'?$
	tcn_input?????????
p 

 
? "??????????
?
,__inference_sequential_layer_call_fn_1154586p!"#$%&'()*+,-./012>?;
4?1
'?$
	tcn_input?????????
p

 
? "??????????
?
,__inference_sequential_layer_call_fn_1154776m!"#$%&'()*+,-./012;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
,__inference_sequential_layer_call_fn_1154821m!"#$%&'()*+,-./012;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
%__inference_signature_wrapper_1154731?!"#$%&'()*+,-./012C?@
? 
9?6
4
	tcn_input'?$
	tcn_input?????????"-?*
(
dense?
dense?????????
?
@__inference_tcn_layer_call_and_return_conditional_losses_1155639t!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p 
? "%?"
?
0?????????n
? ?
@__inference_tcn_layer_call_and_return_conditional_losses_1155923t!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p
? "%?"
?
0?????????n
? ?
%__inference_tcn_layer_call_fn_1155378g!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p 
? "??????????n?
%__inference_tcn_layer_call_fn_1155419g!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p
? "??????????n