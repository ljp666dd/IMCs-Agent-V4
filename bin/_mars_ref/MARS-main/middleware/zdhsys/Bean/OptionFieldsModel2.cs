using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Bean
{
    /// <summary>
    /// 子配方字段
    /// </summary>
    public class OptionFieldsModel2
    {
        /// <summary>
        /// 设备ID
        /// </summary>
        public int Id { get; set; }
        /// <summary>
        /// 设备编号
        /// </summary>
        public string DeviceId { get; set; }
        /// <summary>
        /// 设备标签名
        /// </summary>
        public string TagName { get; set; }
        /// <summary>
        /// 数值
        /// </summary>
        public string TagValue { get; set; }

        /// <summary>
        /// 是否是子配方
        /// </summary>
        public bool IsSub { get; set; }
        /// <summary>
        /// 子配方ID
        /// </summary>
        public int SubId { get; set; }

    }
}
