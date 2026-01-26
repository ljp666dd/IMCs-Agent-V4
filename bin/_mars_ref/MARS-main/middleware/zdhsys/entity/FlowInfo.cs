using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.entity
{
    public class FlowInfo
    {
        /// <summary>
        /// 流程图操作指令
        /// </summary>
        public int Cmd;
        /// <summary>
        /// 流程图ID
        /// </summary>
        public long FlowId;
        /// <summary>
        /// 设备ID
        /// </summary>
        public int DeviceId;
        /// <summary>
        /// 流程图方向名称 -- 绘制直线
        /// </summary>
        public string flowDirName;
        /// <summary>
        /// 背景颜色
        /// </summary>
        public string tempHexBG = "";
        /// <summary>
        /// 边框颜色
        /// </summary>
        public string tempHexBd = "";
        /// <summary>
        /// 字体颜色
        /// </summary>
        public string tempHexTxt = "";
        /// <summary>
        /// 文本内容 -- 设备名称
        /// </summary>
        public string tempTxt = "";
        /// <summary>
        /// 相对偏移坐标X
        /// </summary>
        public double translateTransformX;
        /// <summary>
        /// 相对偏移坐标Y
        /// </summary>
        public double translateTransformY;

        /// <summary>
        /// 流程图操作指令
        /// </summary>
        public int Cmd2;
        /// <summary>
        /// 流程图ID
        /// </summary>
        public long FlowId2;
        /// <summary>
        /// 设备ID
        /// </summary>
        public int DeviceId2;
        /// <summary>
        /// 流程图方向名称 -- 绘制直线
        /// </summary>
        public string flowDirName2;
        /// <summary>
        /// 背景颜色
        /// </summary>
        public string tempHexBG2 = "";
        /// <summary>
        /// 边框颜色
        /// </summary>
        public string tempHexBd2 = "";
        /// <summary>
        /// 字体颜色
        /// </summary>
        public string tempHexTxt2 = "";
        /// <summary>
        /// 文本内容 -- 设备名称
        /// </summary>
        public string tempTxt2 = "";
        /// <summary>
        /// 相对偏移坐标X
        /// </summary>
        public double translateTransformX2;
        /// <summary>
        /// 相对偏移坐标Y
        /// </summary>
        public double translateTransformY2;
    }
}
