using SQLite;
using System.Collections.Generic;

namespace zdhsys.Bean
{
    public class DeviceInfoModel
    {
        /// <summary>
        /// 表ID
        /// </summary>
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }
        /// <summary>
        /// 设备类别 0反应类设备  1转移类设备
        /// </summary>
        public int DeviceType { get; set; }
        /// <summary>
        /// 设备ID --通讯用的
        /// </summary>
        public string DeviceId { get; set; }
        /// <summary>
        /// 所属设备组ID
        /// </summary>
        public string DeviceGroupId { get; set; }
        /// <summary>
        /// 设备名称
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
        /// 设备可操作 --不生成数据库字段
        /// </summary>
        public List<DeviceFieldsModel> Dfms;

        /// <summary>
        /// 设备可操作JSON字符串
        /// </summary>
        public string FieldJson { get; set; }
        /// <summary>
        /// 点位信息
        /// </summary>
        public string PointJson { get; set; }

        /// <summary>
        /// 排序
        /// </summary>
        public int Sort { get; set; }

        #region 下面是其它字段。是从设备上读取上来的。

        /// <summary>
        /// 当前余量
        /// </summary>
        public string Remain { get; set; }

        /// <summary>
        /// 设备状态 1 正常 其它故障
        /// </summary>
        public int DeviceStatus { get; set; }

        /// <summary>
        /// PCR值
        /// </summary>
        public string PCR { get; set; }

        /// <summary>
        /// 浓度
        /// </summary>
        public string Concentration { get; set; }

        /// <summary>
        /// 密度
        /// </summary>
        public string Density { get; set; }

        /// <summary>
        /// 重量
        /// </summary>
        public string Weight { get; set; }

        #endregion
    }
}
