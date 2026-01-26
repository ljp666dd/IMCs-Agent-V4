using SQLite;
using System;
using System.Collections.Generic;

namespace zdhsys.Bean
{
    public class DeviceGroupInfoModel
    {
        /// <summary>
        /// 表ID
        /// </summary>
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }
        /// <summary>
        /// 设备类别
        /// </summary>
        public int DeviceType { get; set; }
        /// <summary>
        /// 设备组ID --通讯用的
        /// </summary>
        public string DeviceId { get; set; }
        /// <summary>
        /// 设备组名称
        /// </summary>
        public string DeviceName { get; set; }
        /// <summary>
        /// 标签名称
        /// </summary>
        public string TagName { get; set; }
        /// <summary>
        /// 标签单位
        /// </summary>
        public string TagUnit { get; set; }
        /// <summary>
        /// 坐标位置:X
        /// </summary>
        public float X { get; set; }
        /// <summary>
        /// 坐标位置:Y
        /// </summary>
        public float Y { get; set; }
        /// <summary>
        /// 坐标位置:Z
        /// </summary>
        public float Z { get; set; }
        /// <summary>
        /// 设备大小:长
        /// </summary>
        public float L { get; set; }
        /// <summary>
        /// 设备大小:宽
        /// </summary>
        public float W { get; set; }
        /// <summary>
        /// 设备大小:高
        /// </summary>
        public float H { get; set; }

        /// <summary>
        /// 操作物体种类:大瓶子0  小瓶子1
        /// </summary>
        public int Bottole { get; set; }

        /// <summary>
        /// 创建时间
        /// </summary>
        public long CreateTime { get; set; }

        /// <summary>
        /// 设备可操作
        /// </summary>
        public List<DeviceFieldsModel> Dfms;

        /// <summary>
        /// 设备可操作JSON字符串
        /// </summary>
        public string FieldJson { get; set; }

        /// <summary>
        /// 当前余量
        /// </summary>
        public string Remain { get; set; }

        /// <summary>
        /// 设备状态 1 正常 其它故障
        /// </summary>
        public int DeviceStatus { get; set; }
        /// <summary>
        /// 通信协议 =GlobalEnum.UnitDeviceType
        /// </summary>
        public string Protocol { get; set; }

    }
}
