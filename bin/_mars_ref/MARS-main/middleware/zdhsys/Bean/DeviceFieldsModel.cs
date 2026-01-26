using SQLite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Bean
{
    public class DeviceFieldsModel
    {
        //[PrimaryKey, AutoIncrement]
        public int Id { get; set; }
        /// <summary>
        /// 字段名称
        /// </summary>
        public string FieldsName { get; set; }
        /// <summary>
        /// 字段内容
        /// </summary>
        public string FieldsContent { get; set; }

        ///// <summary>
        ///// 所属设备
        ///// </summary>
        //public int DeviceId { get; set; }

        ///// <summary>
        ///// 所属组设备
        ///// </summary>
        //public int DeviceGroupId { get; set; }
    }
}
