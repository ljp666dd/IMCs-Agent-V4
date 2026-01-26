using SQLite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.Bean
{
    public class RobotInfoModel
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }
        /// <summary>
        /// 机械臂IP
        /// </summary>
        public string RobotIP { get; set; }
        /// <summary>
        /// 机械臂PORT
        /// </summary>
        public string RobotPort { get; set; }
    }
}
