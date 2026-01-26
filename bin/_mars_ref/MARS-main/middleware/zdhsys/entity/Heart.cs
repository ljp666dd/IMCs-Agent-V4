namespace zdhsys.entity
{
    public class Heart
    {
        /// <summary>
        /// 当前所处操作位置:从0到50的整数
        /// </summary>
        public int ID;
        /// <summary>
        /// 导轨位置:导轨位置，脉冲计数
        /// </summary>
        public int guideway;
        /// <summary>
        /// 末端点位置:Xyz坐标
        /// </summary>
        public float endpoint;
        /// <summary>
        /// 机器人执行状态:0空闲 1执行中
        /// </summary>
        public int rob_sta;
        /// <summary>
        /// 夹爪是否有瓶子:0 无 1有瓶子 2 错误
        /// </summary>
        public int pod_sta;
    }
}
