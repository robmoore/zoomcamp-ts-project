??(
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
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??#
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:d
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
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
shape:d*5
shared_name&$tcn/residual_block_0/conv1D_0/kernel
?
8tcn/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_0/kernel*"
_output_shapes
:d*
dtype0
?
"tcn/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_0/conv1D_0/bias
?
6tcn/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_0/conv1D_0/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_0/conv1D_1/kernel
?
8tcn/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_1/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_0/conv1D_1/bias
?
6tcn/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_0/conv1D_1/bias*
_output_shapes
:d*
dtype0
?
+tcn/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*<
shared_name-+tcn/residual_block_0/matching_conv1D/kernel
?
?tcn/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp+tcn/residual_block_0/matching_conv1D/kernel*"
_output_shapes
:d*
dtype0
?
)tcn/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)tcn/residual_block_0/matching_conv1D/bias
?
=tcn/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp)tcn/residual_block_0/matching_conv1D/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_1/conv1D_0/kernel
?
8tcn/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_0/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_1/conv1D_0/bias
?
6tcn/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_1/conv1D_0/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_1/conv1D_1/kernel
?
8tcn/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_1/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_1/conv1D_1/bias
?
6tcn/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_1/conv1D_1/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_2/conv1D_0/kernel
?
8tcn/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_0/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_2/conv1D_0/bias
?
6tcn/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_2/conv1D_0/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_2/conv1D_1/kernel
?
8tcn/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_1/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_2/conv1D_1/bias
?
6tcn/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_2/conv1D_1/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_3/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_3/conv1D_0/kernel
?
8tcn/residual_block_3/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_0/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_3/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_3/conv1D_0/bias
?
6tcn/residual_block_3/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_3/conv1D_0/bias*
_output_shapes
:d*
dtype0
?
$tcn/residual_block_3/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*5
shared_name&$tcn/residual_block_3/conv1D_1/kernel
?
8tcn/residual_block_3/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_1/kernel*"
_output_shapes
:dd*
dtype0
?
"tcn/residual_block_3/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"tcn/residual_block_3/conv1D_1/bias
?
6tcn/residual_block_3/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_3/conv1D_1/bias*
_output_shapes
:d*
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
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:d
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:
*
dtype0
?
+Adam/tcn/residual_block_0/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*<
shared_name-+Adam/tcn/residual_block_0/conv1D_0/kernel/m
?
?Adam/tcn/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_0/kernel/m*"
_output_shapes
:d*
dtype0
?
)Adam/tcn/residual_block_0/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_0/conv1D_0/bias/m
?
=Adam/tcn/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_0/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_0/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_0/conv1D_1/kernel/m
?
?Adam/tcn/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_1/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_0/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_0/conv1D_1/bias/m
?
=Adam/tcn/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_1/bias/m*
_output_shapes
:d*
dtype0
?
2Adam/tcn/residual_block_0/matching_conv1D/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*C
shared_name42Adam/tcn/residual_block_0/matching_conv1D/kernel/m
?
FAdam/tcn/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/tcn/residual_block_0/matching_conv1D/kernel/m*"
_output_shapes
:d*
dtype0
?
0Adam/tcn/residual_block_0/matching_conv1D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*A
shared_name20Adam/tcn/residual_block_0/matching_conv1D/bias/m
?
DAdam/tcn/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOpReadVariableOp0Adam/tcn/residual_block_0/matching_conv1D/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_1/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_1/conv1D_0/kernel/m
?
?Adam/tcn/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_0/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_1/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_1/conv1D_0/bias/m
?
=Adam/tcn/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_0/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_1/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_1/conv1D_1/kernel/m
?
?Adam/tcn/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_1/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_1/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_1/conv1D_1/bias/m
?
=Adam/tcn/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_1/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_2/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_2/conv1D_0/kernel/m
?
?Adam/tcn/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_0/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_2/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_2/conv1D_0/bias/m
?
=Adam/tcn/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_0/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_2/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_2/conv1D_1/kernel/m
?
?Adam/tcn/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_1/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_2/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_2/conv1D_1/bias/m
?
=Adam/tcn/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_1/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_3/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_3/conv1D_0/kernel/m
?
?Adam/tcn/residual_block_3/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_0/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_3/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_3/conv1D_0/bias/m
?
=Adam/tcn/residual_block_3/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_0/bias/m*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_3/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_3/conv1D_1/kernel/m
?
?Adam/tcn/residual_block_3/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_1/kernel/m*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_3/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_3/conv1D_1/bias/m
?
=Adam/tcn/residual_block_3/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_1/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:d
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:
*
dtype0
?
+Adam/tcn/residual_block_0/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*<
shared_name-+Adam/tcn/residual_block_0/conv1D_0/kernel/v
?
?Adam/tcn/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_0/kernel/v*"
_output_shapes
:d*
dtype0
?
)Adam/tcn/residual_block_0/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_0/conv1D_0/bias/v
?
=Adam/tcn/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_0/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_0/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_0/conv1D_1/kernel/v
?
?Adam/tcn/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_1/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_0/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_0/conv1D_1/bias/v
?
=Adam/tcn/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_1/bias/v*
_output_shapes
:d*
dtype0
?
2Adam/tcn/residual_block_0/matching_conv1D/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*C
shared_name42Adam/tcn/residual_block_0/matching_conv1D/kernel/v
?
FAdam/tcn/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/tcn/residual_block_0/matching_conv1D/kernel/v*"
_output_shapes
:d*
dtype0
?
0Adam/tcn/residual_block_0/matching_conv1D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*A
shared_name20Adam/tcn/residual_block_0/matching_conv1D/bias/v
?
DAdam/tcn/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOpReadVariableOp0Adam/tcn/residual_block_0/matching_conv1D/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_1/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_1/conv1D_0/kernel/v
?
?Adam/tcn/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_0/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_1/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_1/conv1D_0/bias/v
?
=Adam/tcn/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_0/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_1/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_1/conv1D_1/kernel/v
?
?Adam/tcn/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_1/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_1/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_1/conv1D_1/bias/v
?
=Adam/tcn/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_1/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_2/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_2/conv1D_0/kernel/v
?
?Adam/tcn/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_0/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_2/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_2/conv1D_0/bias/v
?
=Adam/tcn/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_0/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_2/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_2/conv1D_1/kernel/v
?
?Adam/tcn/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_1/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_2/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_2/conv1D_1/bias/v
?
=Adam/tcn/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_1/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_3/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_3/conv1D_0/kernel/v
?
?Adam/tcn/residual_block_3/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_0/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_3/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_3/conv1D_0/bias/v
?
=Adam/tcn/residual_block_3/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_0/bias/v*
_output_shapes
:d*
dtype0
?
+Adam/tcn/residual_block_3/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:dd*<
shared_name-+Adam/tcn/residual_block_3/conv1D_1/kernel/v
?
?Adam/tcn/residual_block_3/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_1/kernel/v*"
_output_shapes
:dd*
dtype0
?
)Adam/tcn/residual_block_3/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)Adam/tcn/residual_block_3/conv1D_1/bias/v
?
=Adam/tcn/residual_block_3/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_1/bias/v*
_output_shapes
:d*
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
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/tcn/residual_block_0/matching_conv1D/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/tcn/residual_block_0/matching_conv1D/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_0/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_0/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_1/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_1/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/tcn/residual_block_0/matching_conv1D/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/tcn/residual_block_0/matching_conv1D/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_0/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_0/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_1/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_1/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_tcn_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_tcn_input$tcn/residual_block_0/conv1D_0/kernel"tcn/residual_block_0/conv1D_0/bias$tcn/residual_block_0/conv1D_1/kernel"tcn/residual_block_0/conv1D_1/bias+tcn/residual_block_0/matching_conv1D/kernel)tcn/residual_block_0/matching_conv1D/bias$tcn/residual_block_1/conv1D_0/kernel"tcn/residual_block_1/conv1D_0/bias$tcn/residual_block_1/conv1D_1/kernel"tcn/residual_block_1/conv1D_1/bias$tcn/residual_block_2/conv1D_0/kernel"tcn/residual_block_2/conv1D_0/bias$tcn/residual_block_2/conv1D_1/kernel"tcn/residual_block_2/conv1D_1/bias$tcn/residual_block_3/conv1D_0/kernel"tcn/residual_block_3/conv1D_0/bias$tcn/residual_block_3/conv1D_1/kernel"tcn/residual_block_3/conv1D_1/biasdense_2/kerneldense_2/bias* 
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
%__inference_signature_wrapper_1150821
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp8tcn/residual_block_0/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_0/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_0/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_0/conv1D_1/bias/Read/ReadVariableOp?tcn/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOp=tcn/residual_block_0/matching_conv1D/bias/Read/ReadVariableOp8tcn/residual_block_1/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_1/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_1/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_1/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_2/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_2/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_2/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_2/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_3/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_3/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_3/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_3/conv1D_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpFAdam/tcn/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpDAdam/tcn/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpFAdam/tcn/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpDAdam/tcn/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_1/bias/v/Read/ReadVariableOpConst*T
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
 __inference__traced_save_1152708
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate$tcn/residual_block_0/conv1D_0/kernel"tcn/residual_block_0/conv1D_0/bias$tcn/residual_block_0/conv1D_1/kernel"tcn/residual_block_0/conv1D_1/bias+tcn/residual_block_0/matching_conv1D/kernel)tcn/residual_block_0/matching_conv1D/bias$tcn/residual_block_1/conv1D_0/kernel"tcn/residual_block_1/conv1D_0/bias$tcn/residual_block_1/conv1D_1/kernel"tcn/residual_block_1/conv1D_1/bias$tcn/residual_block_2/conv1D_0/kernel"tcn/residual_block_2/conv1D_0/bias$tcn/residual_block_2/conv1D_1/kernel"tcn/residual_block_2/conv1D_1/bias$tcn/residual_block_3/conv1D_0/kernel"tcn/residual_block_3/conv1D_0/bias$tcn/residual_block_3/conv1D_1/kernel"tcn/residual_block_3/conv1D_1/biastotalcounttotal_1count_1total_2count_2Adam/dense_2/kernel/mAdam/dense_2/bias/m+Adam/tcn/residual_block_0/conv1D_0/kernel/m)Adam/tcn/residual_block_0/conv1D_0/bias/m+Adam/tcn/residual_block_0/conv1D_1/kernel/m)Adam/tcn/residual_block_0/conv1D_1/bias/m2Adam/tcn/residual_block_0/matching_conv1D/kernel/m0Adam/tcn/residual_block_0/matching_conv1D/bias/m+Adam/tcn/residual_block_1/conv1D_0/kernel/m)Adam/tcn/residual_block_1/conv1D_0/bias/m+Adam/tcn/residual_block_1/conv1D_1/kernel/m)Adam/tcn/residual_block_1/conv1D_1/bias/m+Adam/tcn/residual_block_2/conv1D_0/kernel/m)Adam/tcn/residual_block_2/conv1D_0/bias/m+Adam/tcn/residual_block_2/conv1D_1/kernel/m)Adam/tcn/residual_block_2/conv1D_1/bias/m+Adam/tcn/residual_block_3/conv1D_0/kernel/m)Adam/tcn/residual_block_3/conv1D_0/bias/m+Adam/tcn/residual_block_3/conv1D_1/kernel/m)Adam/tcn/residual_block_3/conv1D_1/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/v+Adam/tcn/residual_block_0/conv1D_0/kernel/v)Adam/tcn/residual_block_0/conv1D_0/bias/v+Adam/tcn/residual_block_0/conv1D_1/kernel/v)Adam/tcn/residual_block_0/conv1D_1/bias/v2Adam/tcn/residual_block_0/matching_conv1D/kernel/v0Adam/tcn/residual_block_0/matching_conv1D/bias/v+Adam/tcn/residual_block_1/conv1D_0/kernel/v)Adam/tcn/residual_block_1/conv1D_0/bias/v+Adam/tcn/residual_block_1/conv1D_1/kernel/v)Adam/tcn/residual_block_1/conv1D_1/bias/v+Adam/tcn/residual_block_2/conv1D_0/kernel/v)Adam/tcn/residual_block_2/conv1D_0/bias/v+Adam/tcn/residual_block_2/conv1D_1/kernel/v)Adam/tcn/residual_block_2/conv1D_1/bias/v+Adam/tcn/residual_block_3/conv1D_0/kernel/v)Adam/tcn/residual_block_3/conv1D_0/bias/v+Adam/tcn/residual_block_3/conv1D_1/kernel/v)Adam/tcn/residual_block_3/conv1D_1/bias/v*S
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
#__inference__traced_restore_1152931?? 
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149691

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
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150722
	tcn_input!
tcn_1150679:d
tcn_1150681:d!
tcn_1150683:dd
tcn_1150685:d!
tcn_1150687:d
tcn_1150689:d!
tcn_1150691:dd
tcn_1150693:d!
tcn_1150695:dd
tcn_1150697:d!
tcn_1150699:dd
tcn_1150701:d!
tcn_1150703:dd
tcn_1150705:d!
tcn_1150707:dd
tcn_1150709:d!
tcn_1150711:dd
tcn_1150713:d!
dense_2_1150716:d

dense_2_1150718:

identity??dense_2/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCall	tcn_inputtcn_1150679tcn_1150681tcn_1150683tcn_1150685tcn_1150687tcn_1150689tcn_1150691tcn_1150693tcn_1150695tcn_1150697tcn_1150699tcn_1150701tcn_1150703tcn_1150705tcn_1150707tcn_1150709tcn_1150711tcn_1150713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1149948?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_2_1150716dense_2_1150718*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1149996w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_2/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152287

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152472

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
.__inference_sequential_2_layer_call_fn_1150046
	tcn_input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d

unknown_17:d
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
GPU2*0J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150003o
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
?
e
,__inference_SDropout_0_layer_call_fn_1152186

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149445?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_2_layer_call_fn_1152166

inputs
unknown:d
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
GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1149996o
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
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152191

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149457

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
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149718

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152302

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
?
e
,__inference_SDropout_0_layer_call_fn_1152334

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149601?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_1_layer_call_fn_1152366

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149613v
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
.__inference_sequential_2_layer_call_fn_1150676
	tcn_input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d

unknown_17:d
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
GPU2*0J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150588o
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
?
e
,__inference_SDropout_1_layer_call_fn_1152297

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149562?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_SDropout_0_layer_call_fn_1152255

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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149496v
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
??
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1151499

inputs_
Itcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dK
=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:df
Ptcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:dR
Dtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:d_
Itcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:d_
Itcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:d_
Itcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d
5
'dense_2_biasadd_readvariableop_resource:

identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
*tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_0/PadPadinputs3tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????~
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
:??????????
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
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
:d?
$tcn/residual_block_0/conv1D_0/Conv1DConv2D8tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_0/conv1D_0/BiasAddBiasAdd5tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_0/Act_Conv1D_0/ReluRelu.tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_0/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_0/SDropout_0/dropout/MulMul4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:06tcn/residual_block_0/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_0/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_0/SDropout_0/dropout/random_uniform/shapePack6tcn/residual_block_0/SDropout_0/strided_slice:output:0Gtcn/residual_block_0/SDropout_0/dropout/random_uniform/shape/1:output:08tcn/residual_block_0/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_0/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_0/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_0/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_0/SDropout_0/dropout/GreaterEqualGreaterEqualMtcn/residual_block_0/SDropout_0/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_0/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_0/SDropout_0/dropout/CastCast8tcn/residual_block_0/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_0/SDropout_0/dropout/Mul_1Mul/tcn/residual_block_0/SDropout_0/dropout/Mul:z:00tcn/residual_block_0/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_1/PadPad1tcn/residual_block_0/SDropout_0/dropout/Mul_1:z:03tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d~
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
:?????????d?
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_0/conv1D_1/Conv1DConv2D8tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_0/conv1D_1/BiasAddBiasAdd5tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_0/Act_Conv1D_1/ReluRelu.tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_0/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_0/SDropout_1/dropout/MulMul4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:06tcn/residual_block_0/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_0/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_0/SDropout_1/dropout/random_uniform/shapePack6tcn/residual_block_0/SDropout_1/strided_slice:output:0Gtcn/residual_block_0/SDropout_1/dropout/random_uniform/shape/1:output:08tcn/residual_block_0/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_0/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_0/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_0/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_0/SDropout_1/dropout/GreaterEqualGreaterEqualMtcn/residual_block_0/SDropout_1/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_0/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_0/SDropout_1/dropout/CastCast8tcn/residual_block_0/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_0/SDropout_1/dropout/Mul_1Mul/tcn/residual_block_0/SDropout_1/dropout/Mul:z:00tcn/residual_block_0/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_0/Act_Conv_Blocks/ReluRelu1tcn/residual_block_0/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
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
:d*
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
:d?
+tcn/residual_block_0/matching_conv1D/Conv1DConv2D?tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Atcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
3tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze4tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpDtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
,tcn/residual_block_0/matching_conv1D/BiasAddBiasAdd<tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Ctcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_0/Add_Res/addAddV25tcn/residual_block_0/matching_conv1D/BiasAdd:output:07tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_0/Act_Res_Block/ReluRelu$tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_0/PadPad5tcn/residual_block_0/Act_Res_Block/Relu:activations:03tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	d~
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
:?????????	d?
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_1/conv1D_0/Conv1DConv2D8tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_1/conv1D_0/BiasAddBiasAdd<tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_1/Act_Conv1D_0/ReluRelu.tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_1/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_1/SDropout_0/dropout/MulMul4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:06tcn/residual_block_1/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_1/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_1/SDropout_0/dropout/random_uniform/shapePack6tcn/residual_block_1/SDropout_0/strided_slice:output:0Gtcn/residual_block_1/SDropout_0/dropout/random_uniform/shape/1:output:08tcn/residual_block_1/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_1/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_1/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_1/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_1/SDropout_0/dropout/GreaterEqualGreaterEqualMtcn/residual_block_1/SDropout_0/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_1/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_1/SDropout_0/dropout/CastCast8tcn/residual_block_1/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_1/SDropout_0/dropout/Mul_1Mul/tcn/residual_block_1/SDropout_0/dropout/Mul:z:00tcn/residual_block_1/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_1/PadPad1tcn/residual_block_1/SDropout_0/dropout/Mul_1:z:03tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	d~
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
:?????????	d?
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_1/conv1D_1/Conv1DConv2D8tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_1/conv1D_1/BiasAddBiasAdd<tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_1/Act_Conv1D_1/ReluRelu.tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_1/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_1/SDropout_1/dropout/MulMul4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:06tcn/residual_block_1/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_1/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_1/SDropout_1/dropout/random_uniform/shapePack6tcn/residual_block_1/SDropout_1/strided_slice:output:0Gtcn/residual_block_1/SDropout_1/dropout/random_uniform/shape/1:output:08tcn/residual_block_1/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_1/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_1/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_1/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_1/SDropout_1/dropout/GreaterEqualGreaterEqualMtcn/residual_block_1/SDropout_1/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_1/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_1/SDropout_1/dropout/CastCast8tcn/residual_block_1/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_1/SDropout_1/dropout/Mul_1Mul/tcn/residual_block_1/SDropout_1/dropout/Mul:z:00tcn/residual_block_1/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_1/Act_Conv_Blocks/ReluRelu1tcn/residual_block_1/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_1/Add_Res/addAddV25tcn/residual_block_0/Act_Res_Block/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_1/Act_Res_Block/ReluRelu$tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_0/PadPad5tcn/residual_block_1/Act_Res_Block/Relu:activations:03tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_2/conv1D_0/Conv1DConv2D8tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_2/conv1D_0/BiasAddBiasAdd<tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_2/Act_Conv1D_0/ReluRelu.tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_2/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_2/SDropout_0/dropout/MulMul4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:06tcn/residual_block_2/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_2/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_2/SDropout_0/dropout/random_uniform/shapePack6tcn/residual_block_2/SDropout_0/strided_slice:output:0Gtcn/residual_block_2/SDropout_0/dropout/random_uniform/shape/1:output:08tcn/residual_block_2/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_2/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_2/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_2/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_2/SDropout_0/dropout/GreaterEqualGreaterEqualMtcn/residual_block_2/SDropout_0/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_2/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_2/SDropout_0/dropout/CastCast8tcn/residual_block_2/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_2/SDropout_0/dropout/Mul_1Mul/tcn/residual_block_2/SDropout_0/dropout/Mul:z:00tcn/residual_block_2/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_1/PadPad1tcn/residual_block_2/SDropout_0/dropout/Mul_1:z:03tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_2/conv1D_1/Conv1DConv2D8tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_2/conv1D_1/BiasAddBiasAdd<tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_2/Act_Conv1D_1/ReluRelu.tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_2/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_2/SDropout_1/dropout/MulMul4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:06tcn/residual_block_2/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_2/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_2/SDropout_1/dropout/random_uniform/shapePack6tcn/residual_block_2/SDropout_1/strided_slice:output:0Gtcn/residual_block_2/SDropout_1/dropout/random_uniform/shape/1:output:08tcn/residual_block_2/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_2/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_2/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_2/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_2/SDropout_1/dropout/GreaterEqualGreaterEqualMtcn/residual_block_2/SDropout_1/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_2/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_2/SDropout_1/dropout/CastCast8tcn/residual_block_2/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_2/SDropout_1/dropout/Mul_1Mul/tcn/residual_block_2/SDropout_1/dropout/Mul:z:00tcn/residual_block_2/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_2/Act_Conv_Blocks/ReluRelu1tcn/residual_block_2/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_2/Add_Res/addAddV25tcn/residual_block_1/Act_Res_Block/Relu:activations:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_2/Act_Res_Block/ReluRelu$tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_0/PadPad5tcn/residual_block_2/Act_Res_Block/Relu:activations:03tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_3/conv1D_0/Conv1DConv2D8tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_3/conv1D_0/BiasAddBiasAdd<tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_3/Act_Conv1D_0/ReluRelu.tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_3/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_3/SDropout_0/dropout/MulMul4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:06tcn/residual_block_3/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_3/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_3/SDropout_0/dropout/random_uniform/shapePack6tcn/residual_block_3/SDropout_0/strided_slice:output:0Gtcn/residual_block_3/SDropout_0/dropout/random_uniform/shape/1:output:08tcn/residual_block_3/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_3/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_3/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_3/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_3/SDropout_0/dropout/GreaterEqualGreaterEqualMtcn/residual_block_3/SDropout_0/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_3/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_3/SDropout_0/dropout/CastCast8tcn/residual_block_3/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_3/SDropout_0/dropout/Mul_1Mul/tcn/residual_block_3/SDropout_0/dropout/Mul:z:00tcn/residual_block_3/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_1/PadPad1tcn/residual_block_3/SDropout_0/dropout/Mul_1:z:03tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_3/conv1D_1/Conv1DConv2D8tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_3/conv1D_1/BiasAddBiasAdd<tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_3/Act_Conv1D_1/ReluRelu.tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskr
-tcn/residual_block_3/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+tcn/residual_block_3/SDropout_1/dropout/MulMul4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:06tcn/residual_block_3/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d?
>tcn/residual_block_3/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<tcn/residual_block_3/SDropout_1/dropout/random_uniform/shapePack6tcn/residual_block_3/SDropout_1/strided_slice:output:0Gtcn/residual_block_3/SDropout_1/dropout/random_uniform/shape/1:output:08tcn/residual_block_3/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Dtcn/residual_block_3/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformEtcn/residual_block_3/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0{
6tcn/residual_block_3/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4tcn/residual_block_3/SDropout_1/dropout/GreaterEqualGreaterEqualMtcn/residual_block_3/SDropout_1/dropout/random_uniform/RandomUniform:output:0?tcn/residual_block_3/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
,tcn/residual_block_3/SDropout_1/dropout/CastCast8tcn/residual_block_3/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
-tcn/residual_block_3/SDropout_1/dropout/Mul_1Mul/tcn/residual_block_3/SDropout_1/dropout/Mul:z:00tcn/residual_block_3/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_3/Act_Conv_Blocks/ReluRelu1tcn/residual_block_3/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_3/Add_Res/addAddV25tcn/residual_block_2/Act_Res_Block/Relu:activations:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_3/Act_Res_Block/ReluRelu$tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
tcn/Add_Skip_Connections/addAddV27tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
tcn/Add_Skip_Connections/add_1AddV2 tcn/Add_Skip_Connections/add:z:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
tcn/Add_Skip_Connections/add_2AddV2"tcn/Add_Skip_Connections/add_1:z:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????dy
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
:?????????d*

begin_mask*
end_mask*
shrink_axis_mask?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0?
dense_2/MatMulMatMul'tcn/Slice_Output/strided_slice:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?	
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp5^tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<^tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpH^tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2l
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
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150768
	tcn_input!
tcn_1150725:d
tcn_1150727:d!
tcn_1150729:dd
tcn_1150731:d!
tcn_1150733:d
tcn_1150735:d!
tcn_1150737:dd
tcn_1150739:d!
tcn_1150741:dd
tcn_1150743:d!
tcn_1150745:dd
tcn_1150747:d!
tcn_1150749:dd
tcn_1150751:d!
tcn_1150753:dd
tcn_1150755:d!
tcn_1150757:dd
tcn_1150759:d!
dense_2_1150762:d

dense_2_1150764:

identity??dense_2/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCall	tcn_inputtcn_1150725tcn_1150727tcn_1150729tcn_1150731tcn_1150733tcn_1150735tcn_1150737tcn_1150739tcn_1150741tcn_1150743tcn_1150745tcn_1150747tcn_1150749tcn_1150751tcn_1150753tcn_1150755tcn_1150757tcn_1150759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1150456?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_2_1150762dense_2_1150764*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1149996w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_2/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_1149996

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
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
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152398

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149484

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
,__inference_SDropout_1_layer_call_fn_1152292

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149535v
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149418

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
?
?
.__inference_sequential_2_layer_call_fn_1150911

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d

unknown_17:d
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
GPU2*0J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150588o
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
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_1152176

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
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
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149574

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
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149601

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1151137

inputs_
Itcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dK
=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:df
Ptcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:dR
Dtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:d_
Itcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:d_
Itcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:d_
Itcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:d_
Itcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddK
=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d
5
'dense_2_biasadd_readvariableop_resource:

identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
*tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_0/PadPadinputs3tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????~
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
:??????????
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
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
:d?
$tcn/residual_block_0/conv1D_0/Conv1DConv2D8tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_0/conv1D_0/BiasAddBiasAdd5tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_0/Act_Conv1D_0/ReluRelu.tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_0/SDropout_0/IdentityIdentity4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_0/conv1D_1/PadPad1tcn/residual_block_0/SDropout_0/Identity:output:03tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d~
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
:?????????d?
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_0/conv1D_1/Conv1DConv2D8tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_0/conv1D_1/BiasAddBiasAdd5tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_0/Act_Conv1D_1/ReluRelu.tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_0/SDropout_1/IdentityIdentity4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_0/Act_Conv_Blocks/ReluRelu1tcn/residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
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
:d*
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
:d?
+tcn/residual_block_0/matching_conv1D/Conv1DConv2D?tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Atcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
3tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze4tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpDtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
,tcn/residual_block_0/matching_conv1D/BiasAddBiasAdd<tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Ctcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_0/Add_Res/addAddV25tcn/residual_block_0/matching_conv1D/BiasAdd:output:07tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_0/Act_Res_Block/ReluRelu$tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_0/PadPad5tcn/residual_block_0/Act_Res_Block/Relu:activations:03tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	d~
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
:?????????	d?
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_1/conv1D_0/Conv1DConv2D8tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_1/conv1D_0/BiasAddBiasAdd<tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_1/Act_Conv1D_0/ReluRelu.tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_1/SDropout_0/IdentityIdentity4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_1/conv1D_1/PadPad1tcn/residual_block_1/SDropout_0/Identity:output:03tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	d~
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
:?????????	d?
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_1/conv1D_1/Conv1DConv2D8tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_1/conv1D_1/BiasAddBiasAdd<tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_1/Act_Conv1D_1/ReluRelu.tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_1/SDropout_1/IdentityIdentity4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_1/Act_Conv_Blocks/ReluRelu1tcn/residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_1/Add_Res/addAddV25tcn/residual_block_0/Act_Res_Block/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_1/Act_Res_Block/ReluRelu$tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_0/PadPad5tcn/residual_block_1/Act_Res_Block/Relu:activations:03tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_2/conv1D_0/Conv1DConv2D8tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_2/conv1D_0/BiasAddBiasAdd<tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_2/Act_Conv1D_0/ReluRelu.tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_2/SDropout_0/IdentityIdentity4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_2/conv1D_1/PadPad1tcn/residual_block_2/SDropout_0/Identity:output:03tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_2/conv1D_1/Conv1DConv2D8tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_2/conv1D_1/BiasAddBiasAdd<tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_2/Act_Conv1D_1/ReluRelu.tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_2/SDropout_1/IdentityIdentity4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_2/Act_Conv_Blocks/ReluRelu1tcn/residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_2/Add_Res/addAddV25tcn/residual_block_1/Act_Res_Block/Relu:activations:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_2/Act_Res_Block/ReluRelu$tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_0/PadPad5tcn/residual_block_2/Act_Res_Block/Relu:activations:03tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_3/conv1D_0/Conv1DConv2D8tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_3/conv1D_0/BiasAddBiasAdd<tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_3/Act_Conv1D_0/ReluRelu.tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_3/SDropout_0/IdentityIdentity4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
*tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
!tcn/residual_block_3/conv1D_1/PadPad1tcn/residual_block_3/SDropout_0/Identity:output:03tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d|
2tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Qtcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????d~
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
:?????????d?
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
$tcn/residual_block_3/conv1D_1/Conv1DConv2D8tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
,tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
%tcn/residual_block_3/conv1D_1/BiasAddBiasAdd<tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
&tcn/residual_block_3/Act_Conv1D_1/ReluRelu.tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
(tcn/residual_block_3/SDropout_1/IdentityIdentity4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
)tcn/residual_block_3/Act_Conv_Blocks/ReluRelu1tcn/residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
 tcn/residual_block_3/Add_Res/addAddV25tcn/residual_block_2/Act_Res_Block/Relu:activations:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
'tcn/residual_block_3/Act_Res_Block/ReluRelu$tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
tcn/Add_Skip_Connections/addAddV27tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
tcn/Add_Skip_Connections/add_1AddV2 tcn/Add_Skip_Connections/add:z:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
tcn/Add_Skip_Connections/add_2AddV2"tcn/Add_Skip_Connections/add_1:z:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????dy
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
:?????????d*

begin_mask*
end_mask*
shrink_axis_mask?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0?
dense_2/MatMulMatMul'tcn/Slice_Output/strided_slice:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?	
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp5^tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<^tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpH^tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2l
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
?
%__inference_signature_wrapper_1150821
	tcn_input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d

unknown_17:d
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
"__inference__wrapped_model_1149409o
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
?
e
,__inference_SDropout_1_layer_call_fn_1152445

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149718?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_tcn_layer_call_and_return_conditional_losses_1152157

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:db
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:dN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:d
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
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
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
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
:d?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_0/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_0/SDropout_0/dropout/MulMul0residual_block_0/Act_Conv1D_0/Relu:activations:02residual_block_0/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_0/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_0/SDropout_0/dropout/random_uniform/shapePack2residual_block_0/SDropout_0/strided_slice:output:0Cresidual_block_0/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_0/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_0/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_0/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_0/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_0/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_0/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_0/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_0/SDropout_0/dropout/CastCast4residual_block_0/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_0/SDropout_0/dropout/Mul_1Mul+residual_block_0/SDropout_0/dropout/Mul:z:0,residual_block_0/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/dropout/Mul_1:z:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dz
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
:?????????d?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_0/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_0/SDropout_1/dropout/MulMul0residual_block_0/Act_Conv1D_1/Relu:activations:02residual_block_0/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_0/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_0/SDropout_1/dropout/random_uniform/shapePack2residual_block_0/SDropout_1/strided_slice:output:0Cresidual_block_0/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_0/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_0/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_0/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_0/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_0/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_0/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_0/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_0/SDropout_1/dropout/CastCast4residual_block_0/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_0/SDropout_1/dropout/Mul_1Mul+residual_block_0/SDropout_1/dropout/Mul:z:0,residual_block_0/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
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
:d*
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
:d?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_1/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_1/SDropout_0/dropout/MulMul0residual_block_1/Act_Conv1D_0/Relu:activations:02residual_block_1/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_1/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_1/SDropout_0/dropout/random_uniform/shapePack2residual_block_1/SDropout_0/strided_slice:output:0Cresidual_block_1/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_1/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_1/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_1/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_1/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_1/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_1/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_1/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_1/SDropout_0/dropout/CastCast4residual_block_1/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_1/SDropout_0/dropout/Mul_1Mul+residual_block_1/SDropout_0/dropout/Mul:z:0,residual_block_1/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/dropout/Mul_1:z:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_1/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_1/SDropout_1/dropout/MulMul0residual_block_1/Act_Conv1D_1/Relu:activations:02residual_block_1/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_1/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_1/SDropout_1/dropout/random_uniform/shapePack2residual_block_1/SDropout_1/strided_slice:output:0Cresidual_block_1/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_1/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_1/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_1/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_1/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_1/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_1/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_1/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_1/SDropout_1/dropout/CastCast4residual_block_1/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_1/SDropout_1/dropout/Mul_1Mul+residual_block_1/SDropout_1/dropout/Mul:z:0,residual_block_1/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_2/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_2/SDropout_0/dropout/MulMul0residual_block_2/Act_Conv1D_0/Relu:activations:02residual_block_2/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_2/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_2/SDropout_0/dropout/random_uniform/shapePack2residual_block_2/SDropout_0/strided_slice:output:0Cresidual_block_2/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_2/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_2/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_2/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_2/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_2/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_2/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_2/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_2/SDropout_0/dropout/CastCast4residual_block_2/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_2/SDropout_0/dropout/Mul_1Mul+residual_block_2/SDropout_0/dropout/Mul:z:0,residual_block_2/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/dropout/Mul_1:z:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_2/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_2/SDropout_1/dropout/MulMul0residual_block_2/Act_Conv1D_1/Relu:activations:02residual_block_2/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_2/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_2/SDropout_1/dropout/random_uniform/shapePack2residual_block_2/SDropout_1/strided_slice:output:0Cresidual_block_2/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_2/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_2/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_2/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_2/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_2/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_2/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_2/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_2/SDropout_1/dropout/CastCast4residual_block_2/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_2/SDropout_1/dropout/Mul_1Mul+residual_block_2/SDropout_1/dropout/Mul:z:0,residual_block_2/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_3/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_3/SDropout_0/dropout/MulMul0residual_block_3/Act_Conv1D_0/Relu:activations:02residual_block_3/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_3/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_3/SDropout_0/dropout/random_uniform/shapePack2residual_block_3/SDropout_0/strided_slice:output:0Cresidual_block_3/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_3/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_3/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_3/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_3/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_3/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_3/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_3/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_3/SDropout_0/dropout/CastCast4residual_block_3/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_3/SDropout_0/dropout/Mul_1Mul+residual_block_3/SDropout_0/dropout/Mul:z:0,residual_block_3/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/dropout/Mul_1:z:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_3/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_3/SDropout_1/dropout/MulMul0residual_block_3/Act_Conv1D_1/Relu:activations:02residual_block_3/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_3/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_3/SDropout_1/dropout/random_uniform/shapePack2residual_block_3/SDropout_1/strided_slice:output:0Cresidual_block_3/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_3/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_3/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_3/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_3/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_3/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_3/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_3/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_3/SDropout_1/dropout/CastCast4residual_block_3/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_3/SDropout_1/dropout/Mul_1Mul+residual_block_3/SDropout_1/dropout/Mul:z:0,residual_block_3/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????du
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
:?????????d*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
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
?
?
.__inference_sequential_2_layer_call_fn_1150866

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d

unknown_17:d
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
GPU2*0J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150003o
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
,__inference_SDropout_0_layer_call_fn_1152181

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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149418v
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149496

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
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149679

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149640

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149562

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149652

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
?%
 __inference__traced_save_1152708
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
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
"savev2_count_2_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_1_bias_m_read_readvariableopQ
Msavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_m_read_readvariableopO
Ksavev2_adam_tcn_residual_block_0_matching_conv1d_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_1_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_1_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_1_bias_v_read_readvariableopQ
Msavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_v_read_readvariableopO
Ksavev2_adam_tcn_residual_block_0_matching_conv1d_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_1_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_1_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_1_bias_v_read_readvariableop
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
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop?savev2_tcn_residual_block_0_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_0_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_0_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_0_conv1d_1_bias_read_readvariableopFsavev2_tcn_residual_block_0_matching_conv1d_kernel_read_readvariableopDsavev2_tcn_residual_block_0_matching_conv1d_bias_read_readvariableop?savev2_tcn_residual_block_1_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_1_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_1_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_1_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_2_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_2_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_2_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_2_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_3_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_3_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_3_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_3_conv1d_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_1_bias_m_read_readvariableopMsavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_m_read_readvariableopKsavev2_adam_tcn_residual_block_0_matching_conv1d_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_1_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_1_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_1_bias_v_read_readvariableopMsavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_v_read_readvariableopKsavev2_adam_tcn_residual_block_0_matching_conv1d_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_1_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_1_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :d
:
: : : : : :d:d:dd:d:d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d: : : : : : :d
:
:d:d:dd:d:d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d
:
:d:d:dd:d:d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d
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
:d: 	

_output_shapes
:d:(
$
"
_output_shapes
:dd: 

_output_shapes
:d:($
"
_output_shapes
:d: 

_output_shapes
:d:($
"
_output_shapes
:dd: 

_output_shapes
:d:($
"
_output_shapes
:dd: 

_output_shapes
:d:($
"
_output_shapes
:dd: 

_output_shapes
:d:($
"
_output_shapes
:dd: 

_output_shapes
:d:($
"
_output_shapes
:dd: 

_output_shapes
:d:($
"
_output_shapes
:dd: 

_output_shapes
:d:
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

:d
: !

_output_shapes
:
:("$
"
_output_shapes
:d: #

_output_shapes
:d:($$
"
_output_shapes
:dd: %

_output_shapes
:d:(&$
"
_output_shapes
:d: '

_output_shapes
:d:(($
"
_output_shapes
:dd: )

_output_shapes
:d:(*$
"
_output_shapes
:dd: +

_output_shapes
:d:(,$
"
_output_shapes
:dd: -

_output_shapes
:d:(.$
"
_output_shapes
:dd: /

_output_shapes
:d:(0$
"
_output_shapes
:dd: 1

_output_shapes
:d:(2$
"
_output_shapes
:dd: 3

_output_shapes
:d:$4 

_output_shapes

:d
: 5

_output_shapes
:
:(6$
"
_output_shapes
:d: 7

_output_shapes
:d:(8$
"
_output_shapes
:dd: 9

_output_shapes
:d:(:$
"
_output_shapes
:d: ;

_output_shapes
:d:(<$
"
_output_shapes
:dd: =

_output_shapes
:d:(>$
"
_output_shapes
:dd: ?

_output_shapes
:d:(@$
"
_output_shapes
:dd: A

_output_shapes
:d:(B$
"
_output_shapes
:dd: C

_output_shapes
:d:(D$
"
_output_shapes
:dd: E

_output_shapes
:d:(F$
"
_output_shapes
:dd: G

_output_shapes
:d:H

_output_shapes
: 
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152250

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
,__inference_SDropout_0_layer_call_fn_1152403

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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149652v
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
?
H
,__inference_SDropout_0_layer_call_fn_1152329

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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149574v
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
ď
?
"__inference__wrapped_model_1149409
	tcn_inputl
Vsequential_2_tcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dX
Jsequential_2_tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:ds
]sequential_2_tcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:d_
Qsequential_2_tcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:dl
Vsequential_2_tcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddX
Jsequential_2_tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:dE
3sequential_2_dense_2_matmul_readvariableop_resource:d
B
4sequential_2_dense_2_biasadd_readvariableop_resource:

identity??+sequential_2/dense_2/BiasAdd/ReadVariableOp?*sequential_2/dense_2/MatMul/ReadVariableOp?Asequential_2/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?Hsequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Tsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?Asequential_2/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?Msequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
7sequential_2/tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_0/conv1D_0/PadPad	tcn_input@sequential_2/tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:??????????
@sequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims7sequential_2/tcn/residual_block_0/conv1D_0/Pad:output:0Isequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Msequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype0?
Bsequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d?
1sequential_2/tcn/residual_block_0/conv1D_0/Conv1DConv2DEsequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Asequential_2/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_0/conv1D_0/BiasAddBiasAddBsequential_2/tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0Isequential_2/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_0/Act_Conv1D_0/ReluRelu;sequential_2/tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_0/SDropout_0/IdentityIdentityAsequential_2/tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_0/conv1D_1/PadPad>sequential_2/tcn/residual_block_0/SDropout_0/Identity:output:0@sequential_2/tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
@sequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims7sequential_2/tcn/residual_block_0/conv1D_1/Pad:output:0Isequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
Msequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_0/conv1D_1/Conv1DConv2DEsequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Asequential_2/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_0/conv1D_1/BiasAddBiasAddBsequential_2/tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0Isequential_2/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_0/Act_Conv1D_1/ReluRelu;sequential_2/tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_0/SDropout_1/IdentityIdentityAsequential_2/tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
6sequential_2/tcn/residual_block_0/Act_Conv_Blocks/ReluRelu>sequential_2/tcn/residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
Gsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Csequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDims	tcn_inputPsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Tsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp]sequential_2_tcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype0?
Isequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Esequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDims\sequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Rsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d?
8sequential_2/tcn/residual_block_0/matching_conv1D/Conv1DConv2DLsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Nsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
@sequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueezeAsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Hsequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpQsequential_2_tcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
9sequential_2/tcn/residual_block_0/matching_conv1D/BiasAddBiasAddIsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Psequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
-sequential_2/tcn/residual_block_0/Add_Res/addAddV2Bsequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd:output:0Dsequential_2/tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
4sequential_2/tcn/residual_block_0/Act_Res_Block/ReluRelu1sequential_2/tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_1/conv1D_0/PadPadBsequential_2/tcn/residual_block_0/Act_Res_Block/Relu:activations:0@sequential_2/tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
?sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
[sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Xsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Lsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND7sequential_2/tcn/residual_block_1/conv1D_0/Pad:output:0Usequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Rsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????	d?
@sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDimsIsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0Isequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	d?
Msequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_1/conv1D_0/Conv1DConv2DEsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Lsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDBsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Usequential_2/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Osequential_2/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????d?
Asequential_2/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_1/conv1D_0/BiasAddBiasAddIsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0Isequential_2/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_1/Act_Conv1D_0/ReluRelu;sequential_2/tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_1/SDropout_0/IdentityIdentityAsequential_2/tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_1/conv1D_1/PadPad>sequential_2/tcn/residual_block_1/SDropout_0/Identity:output:0@sequential_2/tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
?sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
[sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Xsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Lsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND7sequential_2/tcn/residual_block_1/conv1D_1/Pad:output:0Usequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Rsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????	d?
@sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDimsIsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0Isequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	d?
Msequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_1/conv1D_1/Conv1DConv2DEsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Lsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDBsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Usequential_2/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Osequential_2/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????d?
Asequential_2/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_1/conv1D_1/BiasAddBiasAddIsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0Isequential_2/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_1/Act_Conv1D_1/ReluRelu;sequential_2/tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_1/SDropout_1/IdentityIdentityAsequential_2/tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
6sequential_2/tcn/residual_block_1/Act_Conv_Blocks/ReluRelu>sequential_2/tcn/residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
-sequential_2/tcn/residual_block_1/Add_Res/addAddV2Bsequential_2/tcn/residual_block_0/Act_Res_Block/Relu:activations:0Dsequential_2/tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
4sequential_2/tcn/residual_block_1/Act_Res_Block/ReluRelu1sequential_2/tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_2/conv1D_0/PadPadBsequential_2/tcn/residual_block_1/Act_Res_Block/Relu:activations:0@sequential_2/tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
?sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
[sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Xsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Lsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND7sequential_2/tcn/residual_block_2/conv1D_0/Pad:output:0Usequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Rsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????d?
@sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDimsIsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0Isequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
Msequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_2/conv1D_0/Conv1DConv2DEsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Lsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDBsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Usequential_2/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Osequential_2/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????d?
Asequential_2/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_2/conv1D_0/BiasAddBiasAddIsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0Isequential_2/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_2/Act_Conv1D_0/ReluRelu;sequential_2/tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_2/SDropout_0/IdentityIdentityAsequential_2/tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_2/conv1D_1/PadPad>sequential_2/tcn/residual_block_2/SDropout_0/Identity:output:0@sequential_2/tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
?sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
[sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Xsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Lsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND7sequential_2/tcn/residual_block_2/conv1D_1/Pad:output:0Usequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Rsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????d?
@sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDimsIsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0Isequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
Msequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_2/conv1D_1/Conv1DConv2DEsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Lsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDBsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Usequential_2/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Osequential_2/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????d?
Asequential_2/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_2/conv1D_1/BiasAddBiasAddIsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0Isequential_2/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_2/Act_Conv1D_1/ReluRelu;sequential_2/tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_2/SDropout_1/IdentityIdentityAsequential_2/tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
6sequential_2/tcn/residual_block_2/Act_Conv_Blocks/ReluRelu>sequential_2/tcn/residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
-sequential_2/tcn/residual_block_2/Add_Res/addAddV2Bsequential_2/tcn/residual_block_1/Act_Res_Block/Relu:activations:0Dsequential_2/tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
4sequential_2/tcn/residual_block_2/Act_Res_Block/ReluRelu1sequential_2/tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_3/conv1D_0/PadPadBsequential_2/tcn/residual_block_2/Act_Res_Block/Relu:activations:0@sequential_2/tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
?sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
[sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Xsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Lsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND7sequential_2/tcn/residual_block_3/conv1D_0/Pad:output:0Usequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Rsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????d?
@sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDimsIsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0Isequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
Msequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_3/conv1D_0/Conv1DConv2DEsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Lsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDBsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Usequential_2/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Osequential_2/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????d?
Asequential_2/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_3/conv1D_0/BiasAddBiasAddIsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0Isequential_2/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_3/Act_Conv1D_0/ReluRelu;sequential_2/tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_3/SDropout_0/IdentityIdentityAsequential_2/tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
7sequential_2/tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
.sequential_2/tcn/residual_block_3/conv1D_1/PadPad>sequential_2/tcn/residual_block_3/SDropout_0/Identity:output:0@sequential_2/tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????d?
?sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
^sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
[sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Xsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
Lsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND7sequential_2/tcn/residual_block_3/conv1D_1/Pad:output:0Usequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Rsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:?????????d?
@sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDimsIsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0Isequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
Msequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpVsequential_2_tcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
dtype0?
Bsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsUsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ksequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:dd?
1sequential_2/tcn/residual_block_3/conv1D_1/Conv1DConv2DEsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0Gsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
9sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze:sequential_2/tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
Lsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ?
@sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDBsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Usequential_2/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Osequential_2/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:?????????d?
Asequential_2/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
2sequential_2/tcn/residual_block_3/conv1D_1/BiasAddBiasAddIsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0Isequential_2/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
3sequential_2/tcn/residual_block_3/Act_Conv1D_1/ReluRelu;sequential_2/tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
5sequential_2/tcn/residual_block_3/SDropout_1/IdentityIdentityAsequential_2/tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
6sequential_2/tcn/residual_block_3/Act_Conv_Blocks/ReluRelu>sequential_2/tcn/residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
-sequential_2/tcn/residual_block_3/Add_Res/addAddV2Bsequential_2/tcn/residual_block_2/Act_Res_Block/Relu:activations:0Dsequential_2/tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
4sequential_2/tcn/residual_block_3/Act_Res_Block/ReluRelu1sequential_2/tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
)sequential_2/tcn/Add_Skip_Connections/addAddV2Dsequential_2/tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0Dsequential_2/tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
+sequential_2/tcn/Add_Skip_Connections/add_1AddV2-sequential_2/tcn/Add_Skip_Connections/add:z:0Dsequential_2/tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
+sequential_2/tcn/Add_Skip_Connections/add_2AddV2/sequential_2/tcn/Add_Skip_Connections/add_1:z:0Dsequential_2/tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
1sequential_2/tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    ?
3sequential_2/tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            ?
3sequential_2/tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
+sequential_2/tcn/Slice_Output/strided_sliceStridedSlice/sequential_2/tcn/Add_Skip_Connections/add_2:z:0:sequential_2/tcn/Slice_Output/strided_slice/stack:output:0<sequential_2/tcn/Slice_Output/strided_slice/stack_1:output:0<sequential_2/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*

begin_mask*
end_mask*
shrink_axis_mask?
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0?
sequential_2/dense_2/MatMulMatMul4sequential_2/tcn/Slice_Output/strided_slice:output:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
t
IdentityIdentity%sequential_2/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOpB^sequential_2/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpI^sequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpU^sequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpB^sequential_2/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpN^sequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2?
Asequential_2/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2?
Hsequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpHsequential_2/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2?
Tsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpTsequential_2/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2?
Asequential_2/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpAsequential_2/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2?
Msequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpMsequential_2/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:V R
+
_output_shapes
:?????????
#
_user_specified_name	tcn_input
?
e
,__inference_SDropout_1_layer_call_fn_1152371

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149640?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
,__inference_SDropout_1_layer_call_fn_1152223

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149484?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152413

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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150588

inputs!
tcn_1150545:d
tcn_1150547:d!
tcn_1150549:dd
tcn_1150551:d!
tcn_1150553:d
tcn_1150555:d!
tcn_1150557:dd
tcn_1150559:d!
tcn_1150561:dd
tcn_1150563:d!
tcn_1150565:dd
tcn_1150567:d!
tcn_1150569:dd
tcn_1150571:d!
tcn_1150573:dd
tcn_1150575:d!
tcn_1150577:dd
tcn_1150579:d!
dense_2_1150582:d

dense_2_1150584:

identity??dense_2/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCallinputstcn_1150545tcn_1150547tcn_1150549tcn_1150551tcn_1150553tcn_1150555tcn_1150557tcn_1150559tcn_1150561tcn_1150563tcn_1150565tcn_1150567tcn_1150569tcn_1150571tcn_1150573tcn_1150575tcn_1150577tcn_1150579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1150456?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_2_1150582dense_2_1150584*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1149996w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_2/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_tcn_layer_call_and_return_conditional_losses_1150456

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:db
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:dN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:d
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
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
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
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
:d?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_0/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_0/SDropout_0/dropout/MulMul0residual_block_0/Act_Conv1D_0/Relu:activations:02residual_block_0/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_0/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_0/SDropout_0/dropout/random_uniform/shapePack2residual_block_0/SDropout_0/strided_slice:output:0Cresidual_block_0/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_0/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_0/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_0/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_0/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_0/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_0/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_0/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_0/SDropout_0/dropout/CastCast4residual_block_0/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_0/SDropout_0/dropout/Mul_1Mul+residual_block_0/SDropout_0/dropout/Mul:z:0,residual_block_0/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/dropout/Mul_1:z:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dz
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
:?????????d?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_0/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_0/SDropout_1/dropout/MulMul0residual_block_0/Act_Conv1D_1/Relu:activations:02residual_block_0/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_0/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_0/SDropout_1/dropout/random_uniform/shapePack2residual_block_0/SDropout_1/strided_slice:output:0Cresidual_block_0/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_0/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_0/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_0/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_0/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_0/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_0/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_0/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_0/SDropout_1/dropout/CastCast4residual_block_0/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_0/SDropout_1/dropout/Mul_1Mul+residual_block_0/SDropout_1/dropout/Mul:z:0,residual_block_0/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
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
:d*
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
:d?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_1/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_1/SDropout_0/dropout/MulMul0residual_block_1/Act_Conv1D_0/Relu:activations:02residual_block_1/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_1/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_1/SDropout_0/dropout/random_uniform/shapePack2residual_block_1/SDropout_0/strided_slice:output:0Cresidual_block_1/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_1/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_1/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_1/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_1/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_1/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_1/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_1/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_1/SDropout_0/dropout/CastCast4residual_block_1/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_1/SDropout_0/dropout/Mul_1Mul+residual_block_1/SDropout_0/dropout/Mul:z:0,residual_block_1/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/dropout/Mul_1:z:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_1/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_1/SDropout_1/dropout/MulMul0residual_block_1/Act_Conv1D_1/Relu:activations:02residual_block_1/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_1/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_1/SDropout_1/dropout/random_uniform/shapePack2residual_block_1/SDropout_1/strided_slice:output:0Cresidual_block_1/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_1/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_1/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_1/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_1/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_1/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_1/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_1/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_1/SDropout_1/dropout/CastCast4residual_block_1/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_1/SDropout_1/dropout/Mul_1Mul+residual_block_1/SDropout_1/dropout/Mul:z:0,residual_block_1/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_2/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_2/SDropout_0/dropout/MulMul0residual_block_2/Act_Conv1D_0/Relu:activations:02residual_block_2/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_2/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_2/SDropout_0/dropout/random_uniform/shapePack2residual_block_2/SDropout_0/strided_slice:output:0Cresidual_block_2/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_2/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_2/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_2/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_2/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_2/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_2/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_2/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_2/SDropout_0/dropout/CastCast4residual_block_2/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_2/SDropout_0/dropout/Mul_1Mul+residual_block_2/SDropout_0/dropout/Mul:z:0,residual_block_2/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/dropout/Mul_1:z:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_2/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_2/SDropout_1/dropout/MulMul0residual_block_2/Act_Conv1D_1/Relu:activations:02residual_block_2/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_2/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_2/SDropout_1/dropout/random_uniform/shapePack2residual_block_2/SDropout_1/strided_slice:output:0Cresidual_block_2/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_2/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_2/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_2/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_2/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_2/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_2/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_2/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_2/SDropout_1/dropout/CastCast4residual_block_2/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_2/SDropout_1/dropout/Mul_1Mul+residual_block_2/SDropout_1/dropout/Mul:z:0,residual_block_2/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_3/SDropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_3/SDropout_0/dropout/MulMul0residual_block_3/Act_Conv1D_0/Relu:activations:02residual_block_3/SDropout_0/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_3/SDropout_0/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_3/SDropout_0/dropout/random_uniform/shapePack2residual_block_3/SDropout_0/strided_slice:output:0Cresidual_block_3/SDropout_0/dropout/random_uniform/shape/1:output:04residual_block_3/SDropout_0/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_3/SDropout_0/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_3/SDropout_0/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_3/SDropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_3/SDropout_0/dropout/GreaterEqualGreaterEqualIresidual_block_3/SDropout_0/dropout/random_uniform/RandomUniform:output:0;residual_block_3/SDropout_0/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_3/SDropout_0/dropout/CastCast4residual_block_3/SDropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_3/SDropout_0/dropout/Mul_1Mul+residual_block_3/SDropout_0/dropout/Mul:z:0,residual_block_3/SDropout_0/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/dropout/Mul_1:z:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
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
shrink_axis_maskn
)residual_block_3/SDropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
'residual_block_3/SDropout_1/dropout/MulMul0residual_block_3/Act_Conv1D_1/Relu:activations:02residual_block_3/SDropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d|
:residual_block_3/SDropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8residual_block_3/SDropout_1/dropout/random_uniform/shapePack2residual_block_3/SDropout_1/strided_slice:output:0Cresidual_block_3/SDropout_1/dropout/random_uniform/shape/1:output:04residual_block_3/SDropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
@residual_block_3/SDropout_1/dropout/random_uniform/RandomUniformRandomUniformAresidual_block_3/SDropout_1/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype0w
2residual_block_3/SDropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
0residual_block_3/SDropout_1/dropout/GreaterEqualGreaterEqualIresidual_block_3/SDropout_1/dropout/random_uniform/RandomUniform:output:0;residual_block_3/SDropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d?
(residual_block_3/SDropout_1/dropout/CastCast4residual_block_3/SDropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d?
)residual_block_3/SDropout_1/dropout/Mul_1Mul+residual_block_3/SDropout_1/dropout/Mul:z:0,residual_block_3/SDropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d?
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????du
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
:?????????d*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
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
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152213

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
@__inference_tcn_layer_call_and_return_conditional_losses_1149948

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:db
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:dN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:d
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
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
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
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
:d?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dz
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
:?????????d?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
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
:d*
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
:d?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????du
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
:?????????d*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
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
?
?
%__inference_tcn_layer_call_fn_1151540

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d
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
:?????????d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1149948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
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
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149535

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
?
e
,__inference_SDropout_0_layer_call_fn_1152260

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149523?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_tcn_layer_call_and_return_conditional_losses_1151801

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:dG
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:db
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:dN
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:d[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:ddG
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:d
identity??0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp?0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp?
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????z
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
:??????????
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
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
:d?
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dz
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
:?????????d?
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
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
:d*
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
:d?
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims

??????????
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????	dz
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
:?????????	d?
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ?
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????dx
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:?
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
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
:?????????dz
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
:?????????d?
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:dd*
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
:dd?
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
?
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????d*
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
:?????????d?
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d?
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d?
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????d?
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????d?
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:?????????du
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
:?????????d*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
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
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149445

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152339

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152228

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
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152361

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152376

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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152265

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152450

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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150003

inputs!
tcn_1149949:d
tcn_1149951:d!
tcn_1149953:dd
tcn_1149955:d!
tcn_1149957:d
tcn_1149959:d!
tcn_1149961:dd
tcn_1149963:d!
tcn_1149965:dd
tcn_1149967:d!
tcn_1149969:dd
tcn_1149971:d!
tcn_1149973:dd
tcn_1149975:d!
tcn_1149977:dd
tcn_1149979:d!
tcn_1149981:dd
tcn_1149983:d!
dense_2_1149997:d

dense_2_1149999:

identity??dense_2/StatefulPartitionedCall?tcn/StatefulPartitionedCall?
tcn/StatefulPartitionedCallStatefulPartitionedCallinputstcn_1149949tcn_1149951tcn_1149953tcn_1149955tcn_1149957tcn_1149959tcn_1149961tcn_1149963tcn_1149965tcn_1149967tcn_1149969tcn_1149971tcn_1149973tcn_1149975tcn_1149977tcn_1149979tcn_1149981tcn_1149983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1149948?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_2_1149997dense_2_1149999*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1149996w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_2/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_tcn_layer_call_fn_1151581

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d 

unknown_11:dd

unknown_12:d 

unknown_13:dd

unknown_14:d 

unknown_15:dd

unknown_16:d
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
:?????????d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_tcn_layer_call_and_return_conditional_losses_1150456o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
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
?
e
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149613

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
?
f
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152324

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149523

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
,__inference_SDropout_1_layer_call_fn_1152218

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149457v
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
?4
#__inference__traced_restore_1152931
file_prefix1
assignvariableop_dense_2_kernel:d
-
assignvariableop_1_dense_2_bias:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: M
7assignvariableop_7_tcn_residual_block_0_conv1d_0_kernel:dC
5assignvariableop_8_tcn_residual_block_0_conv1d_0_bias:dM
7assignvariableop_9_tcn_residual_block_0_conv1d_1_kernel:ddD
6assignvariableop_10_tcn_residual_block_0_conv1d_1_bias:dU
?assignvariableop_11_tcn_residual_block_0_matching_conv1d_kernel:dK
=assignvariableop_12_tcn_residual_block_0_matching_conv1d_bias:dN
8assignvariableop_13_tcn_residual_block_1_conv1d_0_kernel:ddD
6assignvariableop_14_tcn_residual_block_1_conv1d_0_bias:dN
8assignvariableop_15_tcn_residual_block_1_conv1d_1_kernel:ddD
6assignvariableop_16_tcn_residual_block_1_conv1d_1_bias:dN
8assignvariableop_17_tcn_residual_block_2_conv1d_0_kernel:ddD
6assignvariableop_18_tcn_residual_block_2_conv1d_0_bias:dN
8assignvariableop_19_tcn_residual_block_2_conv1d_1_kernel:ddD
6assignvariableop_20_tcn_residual_block_2_conv1d_1_bias:dN
8assignvariableop_21_tcn_residual_block_3_conv1d_0_kernel:ddD
6assignvariableop_22_tcn_residual_block_3_conv1d_0_bias:dN
8assignvariableop_23_tcn_residual_block_3_conv1d_1_kernel:ddD
6assignvariableop_24_tcn_residual_block_3_conv1d_1_bias:d#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: %
assignvariableop_29_total_2: %
assignvariableop_30_count_2: ;
)assignvariableop_31_adam_dense_2_kernel_m:d
5
'assignvariableop_32_adam_dense_2_bias_m:
U
?assignvariableop_33_adam_tcn_residual_block_0_conv1d_0_kernel_m:dK
=assignvariableop_34_adam_tcn_residual_block_0_conv1d_0_bias_m:dU
?assignvariableop_35_adam_tcn_residual_block_0_conv1d_1_kernel_m:ddK
=assignvariableop_36_adam_tcn_residual_block_0_conv1d_1_bias_m:d\
Fassignvariableop_37_adam_tcn_residual_block_0_matching_conv1d_kernel_m:dR
Dassignvariableop_38_adam_tcn_residual_block_0_matching_conv1d_bias_m:dU
?assignvariableop_39_adam_tcn_residual_block_1_conv1d_0_kernel_m:ddK
=assignvariableop_40_adam_tcn_residual_block_1_conv1d_0_bias_m:dU
?assignvariableop_41_adam_tcn_residual_block_1_conv1d_1_kernel_m:ddK
=assignvariableop_42_adam_tcn_residual_block_1_conv1d_1_bias_m:dU
?assignvariableop_43_adam_tcn_residual_block_2_conv1d_0_kernel_m:ddK
=assignvariableop_44_adam_tcn_residual_block_2_conv1d_0_bias_m:dU
?assignvariableop_45_adam_tcn_residual_block_2_conv1d_1_kernel_m:ddK
=assignvariableop_46_adam_tcn_residual_block_2_conv1d_1_bias_m:dU
?assignvariableop_47_adam_tcn_residual_block_3_conv1d_0_kernel_m:ddK
=assignvariableop_48_adam_tcn_residual_block_3_conv1d_0_bias_m:dU
?assignvariableop_49_adam_tcn_residual_block_3_conv1d_1_kernel_m:ddK
=assignvariableop_50_adam_tcn_residual_block_3_conv1d_1_bias_m:d;
)assignvariableop_51_adam_dense_2_kernel_v:d
5
'assignvariableop_52_adam_dense_2_bias_v:
U
?assignvariableop_53_adam_tcn_residual_block_0_conv1d_0_kernel_v:dK
=assignvariableop_54_adam_tcn_residual_block_0_conv1d_0_bias_v:dU
?assignvariableop_55_adam_tcn_residual_block_0_conv1d_1_kernel_v:ddK
=assignvariableop_56_adam_tcn_residual_block_0_conv1d_1_bias_v:d\
Fassignvariableop_57_adam_tcn_residual_block_0_matching_conv1d_kernel_v:dR
Dassignvariableop_58_adam_tcn_residual_block_0_matching_conv1d_bias_v:dU
?assignvariableop_59_adam_tcn_residual_block_1_conv1d_0_kernel_v:ddK
=assignvariableop_60_adam_tcn_residual_block_1_conv1d_0_bias_v:dU
?assignvariableop_61_adam_tcn_residual_block_1_conv1d_1_kernel_v:ddK
=assignvariableop_62_adam_tcn_residual_block_1_conv1d_1_bias_v:dU
?assignvariableop_63_adam_tcn_residual_block_2_conv1d_0_kernel_v:ddK
=assignvariableop_64_adam_tcn_residual_block_2_conv1d_0_bias_v:dU
?assignvariableop_65_adam_tcn_residual_block_2_conv1d_1_kernel_v:ddK
=assignvariableop_66_adam_tcn_residual_block_2_conv1d_1_bias_v:dU
?assignvariableop_67_adam_tcn_residual_block_3_conv1d_0_kernel_v:ddK
=assignvariableop_68_adam_tcn_residual_block_3_conv1d_0_bias_v:dU
?assignvariableop_69_adam_tcn_residual_block_3_conv1d_1_kernel_v:ddK
=assignvariableop_70_adam_tcn_residual_block_3_conv1d_1_bias_v:d
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
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp?assignvariableop_33_adam_tcn_residual_block_0_conv1d_0_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp=assignvariableop_34_adam_tcn_residual_block_0_conv1d_0_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp?assignvariableop_35_adam_tcn_residual_block_0_conv1d_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp=assignvariableop_36_adam_tcn_residual_block_0_conv1d_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpFassignvariableop_37_adam_tcn_residual_block_0_matching_conv1d_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpDassignvariableop_38_adam_tcn_residual_block_0_matching_conv1d_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp?assignvariableop_39_adam_tcn_residual_block_1_conv1d_0_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp=assignvariableop_40_adam_tcn_residual_block_1_conv1d_0_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp?assignvariableop_41_adam_tcn_residual_block_1_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp=assignvariableop_42_adam_tcn_residual_block_1_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp?assignvariableop_43_adam_tcn_residual_block_2_conv1d_0_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp=assignvariableop_44_adam_tcn_residual_block_2_conv1d_0_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_tcn_residual_block_2_conv1d_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp=assignvariableop_46_adam_tcn_residual_block_2_conv1d_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp?assignvariableop_47_adam_tcn_residual_block_3_conv1d_0_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp=assignvariableop_48_adam_tcn_residual_block_3_conv1d_0_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp?assignvariableop_49_adam_tcn_residual_block_3_conv1d_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp=assignvariableop_50_adam_tcn_residual_block_3_conv1d_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp?assignvariableop_53_adam_tcn_residual_block_0_conv1d_0_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp=assignvariableop_54_adam_tcn_residual_block_0_conv1d_0_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp?assignvariableop_55_adam_tcn_residual_block_0_conv1d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp=assignvariableop_56_adam_tcn_residual_block_0_conv1d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpFassignvariableop_57_adam_tcn_residual_block_0_matching_conv1d_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpDassignvariableop_58_adam_tcn_residual_block_0_matching_conv1d_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp?assignvariableop_59_adam_tcn_residual_block_1_conv1d_0_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp=assignvariableop_60_adam_tcn_residual_block_1_conv1d_0_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp?assignvariableop_61_adam_tcn_residual_block_1_conv1d_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp=assignvariableop_62_adam_tcn_residual_block_1_conv1d_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp?assignvariableop_63_adam_tcn_residual_block_2_conv1d_0_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp=assignvariableop_64_adam_tcn_residual_block_2_conv1d_0_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_tcn_residual_block_2_conv1d_1_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp=assignvariableop_66_adam_tcn_residual_block_2_conv1d_1_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp?assignvariableop_67_adam_tcn_residual_block_3_conv1d_0_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp=assignvariableop_68_adam_tcn_residual_block_3_conv1d_0_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp?assignvariableop_69_adam_tcn_residual_block_3_conv1d_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp=assignvariableop_70_adam_tcn_residual_block_3_conv1d_1_bias_vIdentity_70:output:0"/device:CPU:0*
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
?
e
,__inference_SDropout_0_layer_call_fn_1152408

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1149679?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
f
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152435

inputs
identity?;
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
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
,__inference_SDropout_1_layer_call_fn_1152440

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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1149691v
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
serving_default_tcn_input:0?????????;
dense_20
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
 :d
2dense_2/kernel
:
2dense_2/bias
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
::8d2$tcn/residual_block_0/conv1D_0/kernel
0:.d2"tcn/residual_block_0/conv1D_0/bias
::8dd2$tcn/residual_block_0/conv1D_1/kernel
0:.d2"tcn/residual_block_0/conv1D_1/bias
A:?d2+tcn/residual_block_0/matching_conv1D/kernel
7:5d2)tcn/residual_block_0/matching_conv1D/bias
::8dd2$tcn/residual_block_1/conv1D_0/kernel
0:.d2"tcn/residual_block_1/conv1D_0/bias
::8dd2$tcn/residual_block_1/conv1D_1/kernel
0:.d2"tcn/residual_block_1/conv1D_1/bias
::8dd2$tcn/residual_block_2/conv1D_0/kernel
0:.d2"tcn/residual_block_2/conv1D_0/bias
::8dd2$tcn/residual_block_2/conv1D_1/kernel
0:.d2"tcn/residual_block_2/conv1D_1/bias
::8dd2$tcn/residual_block_3/conv1D_0/kernel
0:.d2"tcn/residual_block_3/conv1D_0/bias
::8dd2$tcn/residual_block_3/conv1D_1/kernel
0:.d2"tcn/residual_block_3/conv1D_1/bias
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
%:#d
2Adam/dense_2/kernel/m
:
2Adam/dense_2/bias/m
?:=d2+Adam/tcn/residual_block_0/conv1D_0/kernel/m
5:3d2)Adam/tcn/residual_block_0/conv1D_0/bias/m
?:=dd2+Adam/tcn/residual_block_0/conv1D_1/kernel/m
5:3d2)Adam/tcn/residual_block_0/conv1D_1/bias/m
F:Dd22Adam/tcn/residual_block_0/matching_conv1D/kernel/m
<::d20Adam/tcn/residual_block_0/matching_conv1D/bias/m
?:=dd2+Adam/tcn/residual_block_1/conv1D_0/kernel/m
5:3d2)Adam/tcn/residual_block_1/conv1D_0/bias/m
?:=dd2+Adam/tcn/residual_block_1/conv1D_1/kernel/m
5:3d2)Adam/tcn/residual_block_1/conv1D_1/bias/m
?:=dd2+Adam/tcn/residual_block_2/conv1D_0/kernel/m
5:3d2)Adam/tcn/residual_block_2/conv1D_0/bias/m
?:=dd2+Adam/tcn/residual_block_2/conv1D_1/kernel/m
5:3d2)Adam/tcn/residual_block_2/conv1D_1/bias/m
?:=dd2+Adam/tcn/residual_block_3/conv1D_0/kernel/m
5:3d2)Adam/tcn/residual_block_3/conv1D_0/bias/m
?:=dd2+Adam/tcn/residual_block_3/conv1D_1/kernel/m
5:3d2)Adam/tcn/residual_block_3/conv1D_1/bias/m
%:#d
2Adam/dense_2/kernel/v
:
2Adam/dense_2/bias/v
?:=d2+Adam/tcn/residual_block_0/conv1D_0/kernel/v
5:3d2)Adam/tcn/residual_block_0/conv1D_0/bias/v
?:=dd2+Adam/tcn/residual_block_0/conv1D_1/kernel/v
5:3d2)Adam/tcn/residual_block_0/conv1D_1/bias/v
F:Dd22Adam/tcn/residual_block_0/matching_conv1D/kernel/v
<::d20Adam/tcn/residual_block_0/matching_conv1D/bias/v
?:=dd2+Adam/tcn/residual_block_1/conv1D_0/kernel/v
5:3d2)Adam/tcn/residual_block_1/conv1D_0/bias/v
?:=dd2+Adam/tcn/residual_block_1/conv1D_1/kernel/v
5:3d2)Adam/tcn/residual_block_1/conv1D_1/bias/v
?:=dd2+Adam/tcn/residual_block_2/conv1D_0/kernel/v
5:3d2)Adam/tcn/residual_block_2/conv1D_0/bias/v
?:=dd2+Adam/tcn/residual_block_2/conv1D_1/kernel/v
5:3d2)Adam/tcn/residual_block_2/conv1D_1/bias/v
?:=dd2+Adam/tcn/residual_block_3/conv1D_0/kernel/v
5:3d2)Adam/tcn/residual_block_3/conv1D_0/bias/v
?:=dd2+Adam/tcn/residual_block_3/conv1D_1/kernel/v
5:3d2)Adam/tcn/residual_block_3/conv1D_1/bias/v
?2?
.__inference_sequential_2_layer_call_fn_1150046
.__inference_sequential_2_layer_call_fn_1150866
.__inference_sequential_2_layer_call_fn_1150911
.__inference_sequential_2_layer_call_fn_1150676?
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1151137
I__inference_sequential_2_layer_call_and_return_conditional_losses_1151499
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150722
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150768?
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
"__inference__wrapped_model_1149409	tcn_input"?
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
%__inference_tcn_layer_call_fn_1151540
%__inference_tcn_layer_call_fn_1151581?
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
@__inference_tcn_layer_call_and_return_conditional_losses_1151801
@__inference_tcn_layer_call_and_return_conditional_losses_1152157?
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
)__inference_dense_2_layer_call_fn_1152166?
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1152176?
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
%__inference_signature_wrapper_1150821	tcn_input"?
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
,__inference_SDropout_0_layer_call_fn_1152181
,__inference_SDropout_0_layer_call_fn_1152186?
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152191
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152213?
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
,__inference_SDropout_1_layer_call_fn_1152218
,__inference_SDropout_1_layer_call_fn_1152223?
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152228
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152250?
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
,__inference_SDropout_0_layer_call_fn_1152255
,__inference_SDropout_0_layer_call_fn_1152260?
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152265
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152287?
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
,__inference_SDropout_1_layer_call_fn_1152292
,__inference_SDropout_1_layer_call_fn_1152297?
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152302
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152324?
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
,__inference_SDropout_0_layer_call_fn_1152329
,__inference_SDropout_0_layer_call_fn_1152334?
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152339
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152361?
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
,__inference_SDropout_1_layer_call_fn_1152366
,__inference_SDropout_1_layer_call_fn_1152371?
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152376
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152398?
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
,__inference_SDropout_0_layer_call_fn_1152403
,__inference_SDropout_0_layer_call_fn_1152408?
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152413
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152435?
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
,__inference_SDropout_1_layer_call_fn_1152440
,__inference_SDropout_1_layer_call_fn_1152445?
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
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152450
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152472?
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
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152191?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152213?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152265?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152287?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152339?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152361?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152413?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_0_layer_call_and_return_conditional_losses_1152435?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
,__inference_SDropout_0_layer_call_fn_1152181{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152186{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152255{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152260{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152329{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152334{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152403{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_0_layer_call_fn_1152408{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152228?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152250?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152302?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152324?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152376?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152398?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152450?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
G__inference_SDropout_1_layer_call_and_return_conditional_losses_1152472?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
,__inference_SDropout_1_layer_call_fn_1152218{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152223{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152292{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152297{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152366{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152371{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152440{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
,__inference_SDropout_1_layer_call_fn_1152445{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
"__inference__wrapped_model_1149409?!"#$%&'()*+,-./0126?3
,?)
'?$
	tcn_input?????????
? "1?.
,
dense_2!?
dense_2?????????
?
D__inference_dense_2_layer_call_and_return_conditional_losses_1152176\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? |
)__inference_dense_2_layer_call_fn_1152166O/?,
%?"
 ?
inputs?????????d
? "??????????
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150722}!"#$%&'()*+,-./012>?;
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1150768}!"#$%&'()*+,-./012>?;
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1151137z!"#$%&'()*+,-./012;?8
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1151499z!"#$%&'()*+,-./012;?8
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
.__inference_sequential_2_layer_call_fn_1150046p!"#$%&'()*+,-./012>?;
4?1
'?$
	tcn_input?????????
p 

 
? "??????????
?
.__inference_sequential_2_layer_call_fn_1150676p!"#$%&'()*+,-./012>?;
4?1
'?$
	tcn_input?????????
p

 
? "??????????
?
.__inference_sequential_2_layer_call_fn_1150866m!"#$%&'()*+,-./012;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
.__inference_sequential_2_layer_call_fn_1150911m!"#$%&'()*+,-./012;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
%__inference_signature_wrapper_1150821?!"#$%&'()*+,-./012C?@
? 
9?6
4
	tcn_input'?$
	tcn_input?????????"1?.
,
dense_2!?
dense_2?????????
?
@__inference_tcn_layer_call_and_return_conditional_losses_1151801t!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p 
? "%?"
?
0?????????d
? ?
@__inference_tcn_layer_call_and_return_conditional_losses_1152157t!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p
? "%?"
?
0?????????d
? ?
%__inference_tcn_layer_call_fn_1151540g!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p 
? "??????????d?
%__inference_tcn_layer_call_fn_1151581g!"#$%&'()*+,-./0127?4
-?*
$?!
inputs?????????
p
? "??????????d