using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace zdhsys.entity
{
    public class RobotHeart
    {
        /// <summary>
        /// 操作指令
        /// </summary>
        public int cmd;
        /// <summary>
        /// 执行数据
        /// </summary>
        public object data;
        /// <summary>
        /// 发送方:server = 0 , client = 1
        /// </summary>
        public int point;
    }

}
