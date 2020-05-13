
"use strict";

let AprilTagExtended = require('./AprilTagExtended.js');
let EncoderStamped = require('./EncoderStamped.js');
let ObstacleImageDetection = require('./ObstacleImageDetection.js');
let DroneControl = require('./DroneControl.js');
let KinematicsParameters = require('./KinematicsParameters.js');
let StopLineReading = require('./StopLineReading.js');
let StreetNameDetection = require('./StreetNameDetection.js');
let ObstacleImageDetectionList = require('./ObstacleImageDetectionList.js');
let IntersectionPose = require('./IntersectionPose.js');
let Pose2DStamped = require('./Pose2DStamped.js');
let IntersectionPoseImg = require('./IntersectionPoseImg.js');
let Rect = require('./Rect.js');
let ParamTuner = require('./ParamTuner.js');
let ToFStamped = require('./ToFStamped.js');
let ThetaDotSample = require('./ThetaDotSample.js');
let WheelsCmd = require('./WheelsCmd.js');
let DuckiebotLED = require('./DuckiebotLED.js');
let LEDDetectionDebugInfo = require('./LEDDetectionDebugInfo.js');
let KinematicsWeights = require('./KinematicsWeights.js');
let DuckieSensor = require('./DuckieSensor.js');
let Vsample = require('./Vsample.js');
let SegmentList = require('./SegmentList.js');
let ObstacleProjectedDetection = require('./ObstacleProjectedDetection.js');
let LEDDetection = require('./LEDDetection.js');
let WheelsCmdStamped = require('./WheelsCmdStamped.js');
let LightSensor = require('./LightSensor.js');
let TurnIDandType = require('./TurnIDandType.js');
let AprilTagDetection = require('./AprilTagDetection.js');
let IntersectionPoseImgDebug = require('./IntersectionPoseImgDebug.js');
let VehiclePose = require('./VehiclePose.js');
let DroneMode = require('./DroneMode.js');
let LineFollowerStamped = require('./LineFollowerStamped.js');
let VehicleCorners = require('./VehicleCorners.js');
let ObstacleProjectedDetectionList = require('./ObstacleProjectedDetectionList.js');
let AprilTagDetectionArray = require('./AprilTagDetectionArray.js');
let SignalsDetectionETHZ17 = require('./SignalsDetectionETHZ17.js');
let LEDPattern = require('./LEDPattern.js');
let SourceTargetNodes = require('./SourceTargetNodes.js');
let MaintenanceState = require('./MaintenanceState.js');
let CoordinationSignal = require('./CoordinationSignal.js');
let AntiInstagramTransform_CB = require('./AntiInstagramTransform_CB.js');
let StreetNames = require('./StreetNames.js');
let Vector2D = require('./Vector2D.js');
let Segment = require('./Segment.js');
let LanePose = require('./LanePose.js');
let SceneSegments = require('./SceneSegments.js');
let FSMState = require('./FSMState.js');
let Twist2DStamped = require('./Twist2DStamped.js');
let ObstacleType = require('./ObstacleType.js');
let Trajectory = require('./Trajectory.js');
let SignalsDetection = require('./SignalsDetection.js');
let AntiInstagramHealth = require('./AntiInstagramHealth.js');
let LEDDetectionArray = require('./LEDDetectionArray.js');
let WheelsCmdDBV2Stamped = require('./WheelsCmdDBV2Stamped.js');
let LEDInterpreter = require('./LEDInterpreter.js');
let Pixel = require('./Pixel.js');
let CarControl = require('./CarControl.js');
let BoolStamped = require('./BoolStamped.js');
let AprilTagsWithInfos = require('./AprilTagsWithInfos.js');
let AntiInstagramTransform = require('./AntiInstagramTransform.js');
let CoordinationClearance = require('./CoordinationClearance.js');
let TagInfo = require('./TagInfo.js');
let Rects = require('./Rects.js');

module.exports = {
  AprilTagExtended: AprilTagExtended,
  EncoderStamped: EncoderStamped,
  ObstacleImageDetection: ObstacleImageDetection,
  DroneControl: DroneControl,
  KinematicsParameters: KinematicsParameters,
  StopLineReading: StopLineReading,
  StreetNameDetection: StreetNameDetection,
  ObstacleImageDetectionList: ObstacleImageDetectionList,
  IntersectionPose: IntersectionPose,
  Pose2DStamped: Pose2DStamped,
  IntersectionPoseImg: IntersectionPoseImg,
  Rect: Rect,
  ParamTuner: ParamTuner,
  ToFStamped: ToFStamped,
  ThetaDotSample: ThetaDotSample,
  WheelsCmd: WheelsCmd,
  DuckiebotLED: DuckiebotLED,
  LEDDetectionDebugInfo: LEDDetectionDebugInfo,
  KinematicsWeights: KinematicsWeights,
  DuckieSensor: DuckieSensor,
  Vsample: Vsample,
  SegmentList: SegmentList,
  ObstacleProjectedDetection: ObstacleProjectedDetection,
  LEDDetection: LEDDetection,
  WheelsCmdStamped: WheelsCmdStamped,
  LightSensor: LightSensor,
  TurnIDandType: TurnIDandType,
  AprilTagDetection: AprilTagDetection,
  IntersectionPoseImgDebug: IntersectionPoseImgDebug,
  VehiclePose: VehiclePose,
  DroneMode: DroneMode,
  LineFollowerStamped: LineFollowerStamped,
  VehicleCorners: VehicleCorners,
  ObstacleProjectedDetectionList: ObstacleProjectedDetectionList,
  AprilTagDetectionArray: AprilTagDetectionArray,
  SignalsDetectionETHZ17: SignalsDetectionETHZ17,
  LEDPattern: LEDPattern,
  SourceTargetNodes: SourceTargetNodes,
  MaintenanceState: MaintenanceState,
  CoordinationSignal: CoordinationSignal,
  AntiInstagramTransform_CB: AntiInstagramTransform_CB,
  StreetNames: StreetNames,
  Vector2D: Vector2D,
  Segment: Segment,
  LanePose: LanePose,
  SceneSegments: SceneSegments,
  FSMState: FSMState,
  Twist2DStamped: Twist2DStamped,
  ObstacleType: ObstacleType,
  Trajectory: Trajectory,
  SignalsDetection: SignalsDetection,
  AntiInstagramHealth: AntiInstagramHealth,
  LEDDetectionArray: LEDDetectionArray,
  WheelsCmdDBV2Stamped: WheelsCmdDBV2Stamped,
  LEDInterpreter: LEDInterpreter,
  Pixel: Pixel,
  CarControl: CarControl,
  BoolStamped: BoolStamped,
  AprilTagsWithInfos: AprilTagsWithInfos,
  AntiInstagramTransform: AntiInstagramTransform,
  CoordinationClearance: CoordinationClearance,
  TagInfo: TagInfo,
  Rects: Rects,
};
