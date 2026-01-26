import { createRouter, createWebHashHistory } from "vue-router"

import Home from '../view/home/index.vue'
import Login from '../view/login/index.vue'
import Reasoning from '../view/reasoning/index.vue'
import LayoutView from '../layout/index.vue'
const routes = [
    {
        path: '/',
        component: Login,
        meta: {
            title: '登录'
        },
    },
    {
        path: '/home',
        redirect: '/home',
        component: LayoutView,
        meta: {
            title: '主页'
        },
        children: [
            {
                path: '/home',
                component: Home,
                meta: {
                    title: '主页'
                },
            }
        ]
    },
    {
        path: '/reasoning',
        redirect: '/reasoning',
        component: LayoutView,
        meta: {
            title: '推理中'
        },
        children: [
            {
                path: '/reasoning',
                component: Reasoning,
                meta: {
                    title: '推理中'
                },
            }
        ]
    }

]

const router = createRouter({
    history: createWebHashHistory(),
    routes
});


export default router;