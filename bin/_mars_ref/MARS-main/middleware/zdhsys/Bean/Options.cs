using SQLite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Bean
{
    public class Options
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }
        /// <summary>
        /// 配方名称
        /// </summary>
        public string OptionName { get; set; }
        /// <summary>
        /// 创建时间
        /// </summary>
        public long CreateTime { get; set; }
        /// <summary>
        /// 配方状态 0正常 1已作废
        /// </summary>
        public int Status { get; set; }
        /// <summary>
        /// 配方详情
        /// </summary>
        public string OptionJson { get; set; }
        /// <summary>
        /// 使用流程
        /// </summary>
        public int FlowId { get; set; }

    }
}
