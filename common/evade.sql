/*
 Navicat Premium Data Transfer

 Source Server         : localhost_mysql
 Source Server Type    : MySQL
 Source Server Version : 80012
 Source Host           : localhost:3306
 Source Schema         : evade

 Target Server Type    : MySQL
 Target Server Version : 80012
 File Encoding         : 65001

 Date: 16/07/2020 14:32:51
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for cap_location
-- ----------------------------
DROP TABLE IF EXISTS `cap_location`;
CREATE TABLE `cap_location`  (
  `ip` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '摄像头ip',
  `gate_num` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '闸机编号',
  `direction` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '闸机方向：0出站，1进站，2双向',
  `default_direct` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '默认闸机方向，针对双向闸机，跟谁在一起就默认跟谁方向一样',
  `entrance` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出入口：AE、D或B',
  `entrance_direct` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出入口的进出站：0出站，1进站',
  `entrance_gate_num` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '画面中炸鸡编号与真实闸机编号的绑定',
  `displacement` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '位移方向：up（画面中向上走），down（画面中向下走）',
  `passway_area` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '画面中通道检测区域，左上右下',
  `gate_area` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '画面中闸机门检测区域，左上右下',
  `gate_light_area` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '画面中闸机灯检测区域，左上右下',
  `current_image_shape` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '当前画面尺寸：w*h',
  `create_time` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '创建时间',
  `stop_time` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '停用时间',
  `is_enabled` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '是否启用：y已启用；n已停用'
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of cap_location
-- ----------------------------
INSERT INTO `cap_location` VALUES ('10.6.8.181', '0', '2', '0', 'D', '0', '0', 'down', '0_0_155_360', '0_176_118_199', NULL, '640x360', '2020-07-16 14:19:02.868333', NULL, 'y');
INSERT INTO `cap_location` VALUES ('10.6.8.181', '1', '0', '0', 'D', '0', '1', 'down', '43_0_420_360', '249_128_408_183', '176_161_240_199', '640x360', '2020-07-16 14:20:26.446318', NULL, 'y');
INSERT INTO `cap_location` VALUES ('10.6.8.181', '2', '0', '0', 'D', '0', '2', 'down', '460_0_604_360', '488_161_582_202', '421_150_487_189', '640x360', '2020-07-16 14:21:36.412041', NULL, 'y');
INSERT INTO `cap_location` VALUES ('10.6.8.222', '0', '1', '1', 'AE', '1', '0', 'up', '114_0_282_480', '133_176_271_230', '271_187_344_234', '640x480', '2020-07-16 14:30:16.082039', NULL, 'y');
INSERT INTO `cap_location` VALUES ('10.6.8.222', '1', '1', '1', 'AE', '1', '1', 'up', '339_0_504_480', '349_218_489_268', '504_188_558_232', '640x480', '2020-07-16 14:31:28.897080', NULL, 'y');
INSERT INTO `cap_location` VALUES ('10.6.8.222', '2', '1', '1', 'AE', '1', '2', 'up', '521_0_640_480', '558_174_627_224', NULL, '640x480', '2020-07-16 14:32:26.235973', NULL, 'y');

-- ----------------------------
-- Table structure for details_10.6.8.181
-- ----------------------------
DROP TABLE IF EXISTS `details_10.6.8.181`;
CREATE TABLE `details_10.6.8.181`  (
  `curr_time` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '当前时刻，精确到ms',
  `savefile` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '保存文件路径',
  `pass_status` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '通过状态：0正常通过，1涉嫌逃票',
  `read_time` float(10, 5) NULL DEFAULT NULL COMMENT '读取耗时',
  `detect_time` float(10, 5) NULL DEFAULT NULL COMMENT '检测耗时',
  `predicted_class` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '检测类别',
  `score` float(10, 5) NULL DEFAULT NULL COMMENT '得分值',
  `box` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '人头框，左上右下',
  `person_id` int(10) NULL DEFAULT NULL COMMENT '人物id',
  `trackState` int(2) NULL DEFAULT NULL COMMENT '确认状态：1未确认，2已确认',
  `ip` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '摄像机ip',
  `gate_num` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '闸机编号',
  `direction` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '方向：0出站，1进站'
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
