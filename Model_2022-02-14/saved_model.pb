ܞ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28Ȱ
v
Dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameDense1/kernel
o
!Dense1/kernel/Read/ReadVariableOpReadVariableOpDense1/kernel*
_output_shapes

:@*
dtype0
n
Dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameDense1/bias
g
Dense1/bias/Read/ReadVariableOpReadVariableOpDense1/bias*
_output_shapes
:@*
dtype0
w
Dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*
shared_nameDense2/kernel
p
!Dense2/kernel/Read/ReadVariableOpReadVariableOpDense2/kernel*
_output_shapes
:	@?*
dtype0
o
Dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense2/bias
h
Dense2/bias/Read/ReadVariableOpReadVariableOpDense2/bias*
_output_shapes	
:?*
dtype0
w
Dense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_nameDense3/kernel
p
!Dense3/kernel/Read/ReadVariableOpReadVariableOpDense3/kernel*
_output_shapes
:	?@*
dtype0
n
Dense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameDense3/bias
g
Dense3/bias/Read/ReadVariableOpReadVariableOpDense3/bias*
_output_shapes
:@*
dtype0
v
Dense4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameDense4/kernel
o
!Dense4/kernel/Read/ReadVariableOpReadVariableOpDense4/kernel*
_output_shapes

:@*
dtype0
n
Dense4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense4/bias
g
Dense4/bias/Read/ReadVariableOpReadVariableOpDense4/bias*
_output_shapes
:*
dtype0
v
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

:*
dtype0
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
F
0
1
2
3
4
5
6
7
$8
%9
F
0
1
2
3
4
5
6
7
$8
%9
 
?
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
	regularization_losses
 
YW
VARIABLE_VALUEDense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUEDense2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUEDense3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUEDense4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
 	variables
!trainable_variables
"regularization_losses
YW
VARIABLE_VALUEOutput/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEOutput/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
&	variables
'trainable_variables
(regularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
serving_default_InputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputDense1/kernelDense1/biasDense2/kernelDense2/biasDense3/kernelDense3/biasDense4/kernelDense4/biasOutput/kernelOutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference_signature_wrapper_520
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Dense1/kernel/Read/ReadVariableOpDense1/bias/Read/ReadVariableOp!Dense2/kernel/Read/ReadVariableOpDense2/bias/Read/ReadVariableOp!Dense3/kernel/Read/ReadVariableOpDense3/bias/Read/ReadVariableOp!Dense4/kernel/Read/ReadVariableOpDense4/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? *%
f R
__inference__traced_save_801
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense1/kernelDense1/biasDense2/kernelDense2/biasDense3/kernelDense3/biasDense4/kernelDense4/biasOutput/kernelOutput/bias*
Tin
2*
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
GPU 2J 8? *(
f#R!
__inference__traced_restore_841??
?
?
>__inference_model_layer_call_and_return_conditional_losses_387

inputs

dense1_361:@

dense1_363:@

dense2_366:	@?

dense2_368:	?

dense3_371:	?@

dense3_373:@

dense4_376:@

dense4_378:

output_381:

output_383:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?Dense3/StatefulPartitionedCall?Dense4/StatefulPartitionedCall?Output/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputs
dense1_361
dense1_363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense1_layer_call_and_return_conditional_losses_183?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0
dense2_366
dense2_368*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense2_layer_call_and_return_conditional_losses_200?
Dense3/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0
dense3_371
dense3_373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense3_layer_call_and_return_conditional_losses_217?
Dense4/StatefulPartitionedCallStatefulPartitionedCall'Dense3/StatefulPartitionedCall:output:0
dense4_376
dense4_378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense4_layer_call_and_return_conditional_losses_234?
Output/StatefulPartitionedCallStatefulPartitionedCall'Dense4/StatefulPartitionedCall:output:0
output_381
output_383*
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
?__inference_Output_layer_call_and_return_conditional_losses_251v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall^Dense3/StatefulPartitionedCall^Dense4/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2@
Dense3/StatefulPartitionedCallDense3/StatefulPartitionedCall2@
Dense4/StatefulPartitionedCallDense4/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_model_layer_call_fn_570

inputs
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_387o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
!__inference_signature_wrapper_520	
input
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_165o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameInput
?

?
?__inference_Output_layer_call_and_return_conditional_losses_251

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Dense3_layer_call_and_return_conditional_losses_708

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
?__inference_Dense3_layer_call_and_return_conditional_losses_217

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_model_layer_call_and_return_conditional_losses_493	
input

dense1_467:@

dense1_469:@

dense2_472:	@?

dense2_474:	?

dense3_477:	?@

dense3_479:@

dense4_482:@

dense4_484:

output_487:

output_489:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?Dense3/StatefulPartitionedCall?Dense4/StatefulPartitionedCall?Output/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput
dense1_467
dense1_469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense1_layer_call_and_return_conditional_losses_183?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0
dense2_472
dense2_474*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense2_layer_call_and_return_conditional_losses_200?
Dense3/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0
dense3_477
dense3_479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense3_layer_call_and_return_conditional_losses_217?
Dense4/StatefulPartitionedCallStatefulPartitionedCall'Dense3/StatefulPartitionedCall:output:0
dense4_482
dense4_484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense4_layer_call_and_return_conditional_losses_234?
Output/StatefulPartitionedCallStatefulPartitionedCall'Dense4/StatefulPartitionedCall:output:0
output_487
output_489*
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
?__inference_Output_layer_call_and_return_conditional_losses_251v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall^Dense3/StatefulPartitionedCall^Dense4/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2@
Dense3/StatefulPartitionedCallDense3/StatefulPartitionedCall2@
Dense4/StatefulPartitionedCallDense4/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameInput
?

?
?__inference_Output_layer_call_and_return_conditional_losses_748

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
>__inference_model_layer_call_and_return_conditional_losses_648

inputs7
%dense1_matmul_readvariableop_resource:@4
&dense1_biasadd_readvariableop_resource:@8
%dense2_matmul_readvariableop_resource:	@?5
&dense2_biasadd_readvariableop_resource:	?8
%dense3_matmul_readvariableop_resource:	?@4
&dense3_biasadd_readvariableop_resource:@7
%dense4_matmul_readvariableop_resource:@4
&dense4_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??Dense1/BiasAdd/ReadVariableOp?Dense1/MatMul/ReadVariableOp?Dense2/BiasAdd/ReadVariableOp?Dense2/MatMul/ReadVariableOp?Dense3/BiasAdd/ReadVariableOp?Dense3/MatMul/ReadVariableOp?Dense4/BiasAdd/ReadVariableOp?Dense4/MatMul/ReadVariableOp?Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0w
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@^
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
Dense2/MatMulMatMulDense1/Relu:activations:0$Dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Dense2/BiasAddBiasAddDense2/MatMul:product:0%Dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
Dense2/ReluReluDense2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
Dense3/MatMulMatMulDense2/Relu:activations:0$Dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense3/BiasAddBiasAddDense3/MatMul:product:0%Dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@^
Dense3/ReluReluDense3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dense4/MatMulMatMulDense3/Relu:activations:0$Dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Dense4/BiasAddBiasAddDense4/MatMul:product:0%Dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Dense4/ReluReluDense4/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Output/MatMulMatMulDense4/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp^Dense2/BiasAdd/ReadVariableOp^Dense2/MatMul/ReadVariableOp^Dense3/BiasAdd/ReadVariableOp^Dense3/MatMul/ReadVariableOp^Dense4/BiasAdd/ReadVariableOp^Dense4/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp2>
Dense2/BiasAdd/ReadVariableOpDense2/BiasAdd/ReadVariableOp2<
Dense2/MatMul/ReadVariableOpDense2/MatMul/ReadVariableOp2>
Dense3/BiasAdd/ReadVariableOpDense3/BiasAdd/ReadVariableOp2<
Dense3/MatMul/ReadVariableOpDense3/MatMul/ReadVariableOp2>
Dense4/BiasAdd/ReadVariableOpDense4/BiasAdd/ReadVariableOp2<
Dense4/MatMul/ReadVariableOpDense4/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_Dense3_layer_call_fn_697

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense3_layer_call_and_return_conditional_losses_217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_model_layer_call_and_return_conditional_losses_258

inputs

dense1_184:@

dense1_186:@

dense2_201:	@?

dense2_203:	?

dense3_218:	?@

dense3_220:@

dense4_235:@

dense4_237:

output_252:

output_254:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?Dense3/StatefulPartitionedCall?Dense4/StatefulPartitionedCall?Output/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputs
dense1_184
dense1_186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense1_layer_call_and_return_conditional_losses_183?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0
dense2_201
dense2_203*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense2_layer_call_and_return_conditional_losses_200?
Dense3/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0
dense3_218
dense3_220*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense3_layer_call_and_return_conditional_losses_217?
Dense4/StatefulPartitionedCallStatefulPartitionedCall'Dense3/StatefulPartitionedCall:output:0
dense4_235
dense4_237*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense4_layer_call_and_return_conditional_losses_234?
Output/StatefulPartitionedCallStatefulPartitionedCall'Dense4/StatefulPartitionedCall:output:0
output_252
output_254*
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
?__inference_Output_layer_call_and_return_conditional_losses_251v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall^Dense3/StatefulPartitionedCall^Dense4/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2@
Dense3/StatefulPartitionedCallDense3/StatefulPartitionedCall2@
Dense4/StatefulPartitionedCallDense4/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
__inference__wrapped_model_165	
input=
+model_dense1_matmul_readvariableop_resource:@:
,model_dense1_biasadd_readvariableop_resource:@>
+model_dense2_matmul_readvariableop_resource:	@?;
,model_dense2_biasadd_readvariableop_resource:	?>
+model_dense3_matmul_readvariableop_resource:	?@:
,model_dense3_biasadd_readvariableop_resource:@=
+model_dense4_matmul_readvariableop_resource:@:
,model_dense4_biasadd_readvariableop_resource:=
+model_output_matmul_readvariableop_resource::
,model_output_biasadd_readvariableop_resource:
identity??#model/Dense1/BiasAdd/ReadVariableOp?"model/Dense1/MatMul/ReadVariableOp?#model/Dense2/BiasAdd/ReadVariableOp?"model/Dense2/MatMul/ReadVariableOp?#model/Dense3/BiasAdd/ReadVariableOp?"model/Dense3/MatMul/ReadVariableOp?#model/Dense4/BiasAdd/ReadVariableOp?"model/Dense4/MatMul/ReadVariableOp?#model/Output/BiasAdd/ReadVariableOp?"model/Output/MatMul/ReadVariableOp?
"model/Dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model/Dense1/MatMulMatMulinput*model/Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#model/Dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/Dense1/BiasAddBiasAddmodel/Dense1/MatMul:product:0+model/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
model/Dense1/ReluRelumodel/Dense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
"model/Dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
model/Dense2/MatMulMatMulmodel/Dense1/Relu:activations:0*model/Dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#model/Dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/Dense2/BiasAddBiasAddmodel/Dense2/MatMul:product:0+model/Dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
model/Dense2/ReluRelumodel/Dense2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"model/Dense3/MatMul/ReadVariableOpReadVariableOp+model_dense3_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
model/Dense3/MatMulMatMulmodel/Dense2/Relu:activations:0*model/Dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#model/Dense3/BiasAdd/ReadVariableOpReadVariableOp,model_dense3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/Dense3/BiasAddBiasAddmodel/Dense3/MatMul:product:0+model/Dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
model/Dense3/ReluRelumodel/Dense3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
"model/Dense4/MatMul/ReadVariableOpReadVariableOp+model_dense4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model/Dense4/MatMulMatMulmodel/Dense3/Relu:activations:0*model/Dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#model/Dense4/BiasAdd/ReadVariableOpReadVariableOp,model_dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/Dense4/BiasAddBiasAddmodel/Dense4/MatMul:product:0+model/Dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
model/Dense4/ReluRelumodel/Dense4/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"model/Output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model/Output/MatMulMatMulmodel/Dense4/Relu:activations:0*model/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#model/Output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/Output/BiasAddBiasAddmodel/Output/MatMul:product:0+model/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
model/Output/SoftmaxSoftmaxmodel/Output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymodel/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^model/Dense1/BiasAdd/ReadVariableOp#^model/Dense1/MatMul/ReadVariableOp$^model/Dense2/BiasAdd/ReadVariableOp#^model/Dense2/MatMul/ReadVariableOp$^model/Dense3/BiasAdd/ReadVariableOp#^model/Dense3/MatMul/ReadVariableOp$^model/Dense4/BiasAdd/ReadVariableOp#^model/Dense4/MatMul/ReadVariableOp$^model/Output/BiasAdd/ReadVariableOp#^model/Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2J
#model/Dense1/BiasAdd/ReadVariableOp#model/Dense1/BiasAdd/ReadVariableOp2H
"model/Dense1/MatMul/ReadVariableOp"model/Dense1/MatMul/ReadVariableOp2J
#model/Dense2/BiasAdd/ReadVariableOp#model/Dense2/BiasAdd/ReadVariableOp2H
"model/Dense2/MatMul/ReadVariableOp"model/Dense2/MatMul/ReadVariableOp2J
#model/Dense3/BiasAdd/ReadVariableOp#model/Dense3/BiasAdd/ReadVariableOp2H
"model/Dense3/MatMul/ReadVariableOp"model/Dense3/MatMul/ReadVariableOp2J
#model/Dense4/BiasAdd/ReadVariableOp#model/Dense4/BiasAdd/ReadVariableOp2H
"model/Dense4/MatMul/ReadVariableOp"model/Dense4/MatMul/ReadVariableOp2J
#model/Output/BiasAdd/ReadVariableOp#model/Output/BiasAdd/ReadVariableOp2H
"model/Output/MatMul/ReadVariableOp"model/Output/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameInput
?
?
>__inference_model_layer_call_and_return_conditional_losses_464	
input

dense1_438:@

dense1_440:@

dense2_443:	@?

dense2_445:	?

dense3_448:	?@

dense3_450:@

dense4_453:@

dense4_455:

output_458:

output_460:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?Dense3/StatefulPartitionedCall?Dense4/StatefulPartitionedCall?Output/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput
dense1_438
dense1_440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense1_layer_call_and_return_conditional_losses_183?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0
dense2_443
dense2_445*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense2_layer_call_and_return_conditional_losses_200?
Dense3/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0
dense3_448
dense3_450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense3_layer_call_and_return_conditional_losses_217?
Dense4/StatefulPartitionedCallStatefulPartitionedCall'Dense3/StatefulPartitionedCall:output:0
dense4_453
dense4_455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense4_layer_call_and_return_conditional_losses_234?
Output/StatefulPartitionedCallStatefulPartitionedCall'Dense4/StatefulPartitionedCall:output:0
output_458
output_460*
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
?__inference_Output_layer_call_and_return_conditional_losses_251v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall^Dense3/StatefulPartitionedCall^Dense4/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2@
Dense3/StatefulPartitionedCallDense3/StatefulPartitionedCall2@
Dense4/StatefulPartitionedCallDense4/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameInput
?
?
$__inference_Dense1_layer_call_fn_657

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense1_layer_call_and_return_conditional_losses_183o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_801
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop,
(savev2_dense3_kernel_read_readvariableop*
&savev2_dense3_bias_read_readvariableop,
(savev2_dense4_kernel_read_readvariableop*
&savev2_dense4_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop(savev2_dense3_kernel_read_readvariableop&savev2_dense3_bias_read_readvariableop(savev2_dense4_kernel_read_readvariableop&savev2_dense4_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
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

identity_1Identity_1:output:0*j
_input_shapesY
W: :@:@:	@?:?:	?@:@:@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: 
?
?
$__inference_Output_layer_call_fn_737

inputs
unknown:
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
?__inference_Output_layer_call_and_return_conditional_losses_251o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Dense4_layer_call_and_return_conditional_losses_728

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?+
?
>__inference_model_layer_call_and_return_conditional_losses_609

inputs7
%dense1_matmul_readvariableop_resource:@4
&dense1_biasadd_readvariableop_resource:@8
%dense2_matmul_readvariableop_resource:	@?5
&dense2_biasadd_readvariableop_resource:	?8
%dense3_matmul_readvariableop_resource:	?@4
&dense3_biasadd_readvariableop_resource:@7
%dense4_matmul_readvariableop_resource:@4
&dense4_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??Dense1/BiasAdd/ReadVariableOp?Dense1/MatMul/ReadVariableOp?Dense2/BiasAdd/ReadVariableOp?Dense2/MatMul/ReadVariableOp?Dense3/BiasAdd/ReadVariableOp?Dense3/MatMul/ReadVariableOp?Dense4/BiasAdd/ReadVariableOp?Dense4/MatMul/ReadVariableOp?Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0w
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@^
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
Dense2/MatMulMatMulDense1/Relu:activations:0$Dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Dense2/BiasAddBiasAddDense2/MatMul:product:0%Dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
Dense2/ReluReluDense2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
Dense3/MatMulMatMulDense2/Relu:activations:0$Dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense3/BiasAddBiasAddDense3/MatMul:product:0%Dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@^
Dense3/ReluReluDense3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dense4/MatMulMatMulDense3/Relu:activations:0$Dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Dense4/BiasAddBiasAddDense4/MatMul:product:0%Dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Dense4/ReluReluDense4/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Output/MatMulMatMulDense4/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp^Dense2/BiasAdd/ReadVariableOp^Dense2/MatMul/ReadVariableOp^Dense3/BiasAdd/ReadVariableOp^Dense3/MatMul/ReadVariableOp^Dense4/BiasAdd/ReadVariableOp^Dense4/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp2>
Dense2/BiasAdd/ReadVariableOpDense2/BiasAdd/ReadVariableOp2<
Dense2/MatMul/ReadVariableOpDense2/MatMul/ReadVariableOp2>
Dense3/BiasAdd/ReadVariableOpDense3/BiasAdd/ReadVariableOp2<
Dense3/MatMul/ReadVariableOpDense3/MatMul/ReadVariableOp2>
Dense4/BiasAdd/ReadVariableOpDense4/BiasAdd/ReadVariableOp2<
Dense4/MatMul/ReadVariableOpDense4/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_model_layer_call_fn_281	
input
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_258o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameInput
?

?
#__inference_model_layer_call_fn_545

inputs
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_258o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Dense1_layer_call_and_return_conditional_losses_668

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Dense1_layer_call_and_return_conditional_losses_183

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Dense2_layer_call_and_return_conditional_losses_688

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_Dense4_layer_call_fn_717

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense4_layer_call_and_return_conditional_losses_234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
?__inference_Dense4_layer_call_and_return_conditional_losses_234

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?*
?
__inference__traced_restore_841
file_prefix0
assignvariableop_dense1_kernel:@,
assignvariableop_1_dense1_bias:@3
 assignvariableop_2_dense2_kernel:	@?-
assignvariableop_3_dense2_bias:	?3
 assignvariableop_4_dense3_kernel:	?@,
assignvariableop_5_dense3_bias:@2
 assignvariableop_6_dense4_kernel:@,
assignvariableop_7_dense4_bias:2
 assignvariableop_8_output_kernel:,
assignvariableop_9_output_bias:
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
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
$__inference_Dense2_layer_call_fn_677

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense2_layer_call_and_return_conditional_losses_200p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
#__inference_model_layer_call_fn_435	
input
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_387o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameInput
?

?
?__inference_Dense2_layer_call_and_return_conditional_losses_200

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
Input.
serving_default_Input:0?????????:
Output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?]
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
H__call__
*I&call_and_return_all_conditional_losses
J_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
	regularization_losses
H__call__
J_default_save_signature
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
,
Userving_default"
signature_map
:@2Dense1/kernel
:@2Dense1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 :	@?2Dense2/kernel
:?2Dense2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
 :	?@2Dense3/kernel
:@2Dense3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
:@2Dense4/kernel
:2Dense4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
 	variables
!trainable_variables
"regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
:2Output/kernel
:2Output/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
&	variables
'trainable_variables
(regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
?2?
#__inference_model_layer_call_fn_281
#__inference_model_layer_call_fn_545
#__inference_model_layer_call_fn_570
#__inference_model_layer_call_fn_435?
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
>__inference_model_layer_call_and_return_conditional_losses_609
>__inference_model_layer_call_and_return_conditional_losses_648
>__inference_model_layer_call_and_return_conditional_losses_464
>__inference_model_layer_call_and_return_conditional_losses_493?
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
__inference__wrapped_model_165Input"?
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
?2?
$__inference_Dense1_layer_call_fn_657?
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
?__inference_Dense1_layer_call_and_return_conditional_losses_668?
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
$__inference_Dense2_layer_call_fn_677?
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
?__inference_Dense2_layer_call_and_return_conditional_losses_688?
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
$__inference_Dense3_layer_call_fn_697?
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
?__inference_Dense3_layer_call_and_return_conditional_losses_708?
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
$__inference_Dense4_layer_call_fn_717?
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
?__inference_Dense4_layer_call_and_return_conditional_losses_728?
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
$__inference_Output_layer_call_fn_737?
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
?__inference_Output_layer_call_and_return_conditional_losses_748?
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
!__inference_signature_wrapper_520Input"?
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
 ?
?__inference_Dense1_layer_call_and_return_conditional_losses_668\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? w
$__inference_Dense1_layer_call_fn_657O/?,
%?"
 ?
inputs?????????
? "??????????@?
?__inference_Dense2_layer_call_and_return_conditional_losses_688]/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? x
$__inference_Dense2_layer_call_fn_677P/?,
%?"
 ?
inputs?????????@
? "????????????
?__inference_Dense3_layer_call_and_return_conditional_losses_708]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? x
$__inference_Dense3_layer_call_fn_697P0?-
&?#
!?
inputs??????????
? "??????????@?
?__inference_Dense4_layer_call_and_return_conditional_losses_728\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? w
$__inference_Dense4_layer_call_fn_717O/?,
%?"
 ?
inputs?????????@
? "???????????
?__inference_Output_layer_call_and_return_conditional_losses_748\$%/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? w
$__inference_Output_layer_call_fn_737O$%/?,
%?"
 ?
inputs?????????
? "???????????
__inference__wrapped_model_165m
$%.?+
$?!
?
Input?????????
? "/?,
*
Output ?
Output??????????
>__inference_model_layer_call_and_return_conditional_losses_464k
$%6?3
,?)
?
Input?????????
p 

 
? "%?"
?
0?????????
? ?
>__inference_model_layer_call_and_return_conditional_losses_493k
$%6?3
,?)
?
Input?????????
p

 
? "%?"
?
0?????????
? ?
>__inference_model_layer_call_and_return_conditional_losses_609l
$%7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
>__inference_model_layer_call_and_return_conditional_losses_648l
$%7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
#__inference_model_layer_call_fn_281^
$%6?3
,?)
?
Input?????????
p 

 
? "???????????
#__inference_model_layer_call_fn_435^
$%6?3
,?)
?
Input?????????
p

 
? "???????????
#__inference_model_layer_call_fn_545_
$%7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference_model_layer_call_fn_570_
$%7?4
-?*
 ?
inputs?????????
p

 
? "???????????
!__inference_signature_wrapper_520v
$%7?4
? 
-?*
(
Input?
Input?????????"/?,
*
Output ?
Output?????????