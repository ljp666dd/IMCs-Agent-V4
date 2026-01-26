using zdhsys.entity;

namespace zdhsys.Unitils
{
    public class RobotCmd
    {
        /// <summary>
        /// 从目标位置取瓶子
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static RobotHeart TakeBottle(string src)
        {
            // cmd = 2 取瓶子
            RobotHeart rh = new RobotHeart
            {
                cmd = 2,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = int.Parse(src),
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }

        /// <summary>
        /// 将瓶子放到目标位置
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static RobotHeart PutTheBottle(string src)
        {
            // cmd = 3 放瓶子
            RobotHeart rh = new RobotHeart
            {
                cmd = 3,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = 0,
                ID_dst = int.Parse(src)
            };
            rh.data = md;
            return rh;
        }

        /// <summary>
        /// 移动瓶子
        /// </summary>
        /// <param name="src">当前位置</param>
        /// <param name="dst">目标位置</param>
        /// <returns></returns>
        public static RobotHeart MoveBottle(string src, string dst)
        {
            // cmd = 1 移动瓶子
            RobotHeart rh = new RobotHeart
            {
                cmd = 1,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = int.Parse(src),
                ID_dst = int.Parse(dst)
            };
            rh.data = md;
            return rh;
        }

        /// <summary>
        /// A6 接药水操作
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static RobotHeart WaterIntake(string src)
        {
            // cmd = 6 接药水操作
            RobotHeart rh = new RobotHeart
            {
                cmd = 6,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = int.Parse(src),
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }
        /// <summary>
        /// A7 接药水操作_返回
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static RobotHeart WaterReturn(string src)
        {
            // cmd = 7 接药水操作_返回
            RobotHeart rh = new RobotHeart
            {
                cmd = 7,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = int.Parse(src),
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }
        /// <summary>
        /// 机械臂回0（指的是，机械臂收起，但导轨不运动）
        /// </summary>
        /// <returns></returns>
        public static RobotHeart ArmRetraction()
        {
            RobotHeart rh = new RobotHeart
            {
                cmd = 4,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = 0,
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }

        /// <summary>
        /// 去位置ID10时使用，防止碰撞
        /// </summary>
        /// <returns></returns>
        public static RobotHeart ArmReturn10()
        {
            //机械臂回0位置(机械臂收起，朝向平行导轨，去位置ID10时使用，防止碰撞)
            RobotHeart rh = new RobotHeart
            {
                cmd = 10,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = 0,
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }

        /// <summary>
        /// 机械臂收起，同时，导轨运动到0位置
        /// </summary>
        /// <returns></returns>
        public static RobotHeart ArmReturn_0()
        {
            RobotHeart rh = new RobotHeart
            {
                cmd = 5,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = 0,
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }
        /// <summary>
        /// 急停
        /// </summary>
        /// <returns></returns>
        public static RobotHeart ArmScram()
        {
            RobotHeart rh = new RobotHeart
            {
                cmd = 0,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = 0,
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }
        /// <summary>
        /// 软件复位（不要随意使用）
        /// </summary>
        /// <returns></returns>
        public static RobotHeart SoftwareReset()
        {
            RobotHeart rh = new RobotHeart
            {
                cmd = 9,
                point = 1
            };
            MoveData md = new MoveData
            {
                ID_src = 0,
                ID_dst = 0
            };
            rh.data = md;
            return rh;
        }

    }
}
