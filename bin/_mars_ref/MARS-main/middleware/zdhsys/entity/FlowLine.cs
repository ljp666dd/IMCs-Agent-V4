using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using zdhsys.Control;

namespace zdhsys.entity
{
    public class FlowLine
    {
        public FlowButton2 startFlow;
        public string startName;

        public FlowButton2 endFlow;
        public string endName;

        public long id;
    }
}
