using SQLite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Bean
{
    public class FlowModel
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }
        /// <summary>
        /// 流程名称
        /// </summary>
        public string FlowName { get; set; }
        /// <summary>
        /// 流程编号
        /// </summary>
        public string FlowNO { get; set; }
        /// <summary>
        /// 创建时间
        /// </summary>
        public long CreateTime { get; set; }
        /// <summary>
        /// 流程状态 0正常 1禁用
        /// </summary>
        public int Status { get; set; }
        /// <summary>
        /// 流程详情
        /// </summary>
        public string FlowJson { get; set; }
    }
}
