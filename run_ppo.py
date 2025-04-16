import argparse
import genesis as gs
import torch
from algo.ppo_agent import PPOAgent
from env import *
import os
import matplotlib.pyplot as plt


gs.init(backend=gs.gpu, precision="32")

task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv,
    'ShadowHandBase': ShadowHandBaseEnv,
    'GraspRandomBlockCamera' : GraspRandomBlockCamEnv,
    'GraspRandomBlockVel' : GraspRandomBlockVelEnv
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]  
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")

def train_ppo(args):
    if args.load_path == "default":
        load = True
        checkpoint_path = f"logs/{args.task}_ppo_checkpoint_released.pth"
    elif args.load_path: 
        load = True
        checkpoint_path = args.load_path
    else:
        load = False
        checkpoint_path = f"logs/{args.task}_ppo_checkpoint.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created environment: {env}")
    
    agent = PPOAgent(input_dim=env.state_dim, output_dim=env.action_space, lr=1e-3, gamma=0.99, clip_epsilon=0.2, device=args.device, load=load, \
                     num_envs=args.num_envs, hidden_dim=args.hidden_dim, checkpoint_path=checkpoint_path)
    if args.device == "mps":
        gs.tools.run_in_another_thread(fn=run, args=(env, agent))
        env.scene.viewer.start()
    else:
        run(env, agent)

def run(env, agent):
    num_episodes = 4000
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    manual_action_value = 1

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        # threshold = 1.0
        # done_array = torch.tensor([False] * env.num_envs).to(args.device)
        done_array = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
        states, actions, rewards, dones = [], [], [], []
        grasp_rewards = torch.zeros(env.num_envs).to(args.device)

        for step in range(700):

            action = agent.select_action(state)  # Select actions for all environments first

            if step % 5 == 0:
                for i in range(env.num_envs):  # Loop over environments
                    if grasp_rewards[i] == 10:  # Check if grasp reward is 10
                        action[i] = manual_action_value  # Override only this environment's action
                        
            
        
        

            # # Make sure this has shape [num_envs], e.g. tensor([0, 1, 2, 1])
            # if action.dim() == 0:
            #     action = action.unsqueeze(0)
            # Save action
            
            # if not isinstance(action, torch.Tensor):
            #     action = torch.tensor(action, device=env.device)

            # if action.ndim == 0:  # scalar
            #     action = action.unsqueeze(0)
            # Save action
            actions.append(action.clone().detach())
            # Take step in environment
            next_state, reward, done, grasp_rewards = env.step(action)

            states.append(state)
            # actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        agent.train(states, actions, rewards, dones)
        
        if agent.logged_ratios and agent.logged_advantages:  # make sure it's not empty
            ratios = torch.cat(agent.logged_ratios)
            advantages = torch.cat(agent.logged_advantages)
            epsilon = agent.clip_epsilon

            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

            agent.unclipped_log.append(unclipped.mean().item())
            agent.clipped_log.append(clipped.mean().item())


            if episode % 3 == 0 and agent.logged_ratios and agent.logged_advantages:
                # Save scatter plot
                plt.figure(figsize=(10, 6))
                plt.scatter(ratios.numpy(), unclipped.numpy(), label='Unclipped', alpha=0.5, marker='.')
                plt.scatter(ratios.numpy(), clipped.numpy(), label='Clipped', alpha=0.5, marker='.')
                plt.axvline(1 - epsilon, color='gray', linestyle='--', label='Clip Range')
                plt.axvline(1 + epsilon, color='gray', linestyle='--')
                plt.xlabel("Probability Ratio (r)")
                plt.ylabel("Surrogate Objective (r * A)")
                plt.title(f"PPO Clipped Objective - Episode {episode}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                r_min = ratios.min().item()
                r_max = ratios.max().item()
                plt.xlim(r_min - 0.01, r_max + 0.01)

                plt.savefig(f"logs/ppo_objective_ep{episode}.png")
                print("r range:", ratios.min().item(), "to", ratios.max().item())

                plt.close()

                # Loss curve
                if agent.loss_log:
                    plt.figure(figsize=(10, 4))
                    plt.plot(agent.loss_log, label="PPO Loss")
                    plt.xlabel("Update Step")
                    plt.ylabel("Loss")
                    plt.title("PPO Policy Loss Over Time")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"logs/ppo_loss_curve_ep{episode}.png")
                    plt.close()

                # Clipped vs Unclipped over time
                plt.plot(agent.unclipped_log, label="Unclipped Objective")
                plt.plot(agent.clipped_log, label="Clipped Objective")
                plt.xlabel("Update Step")
                plt.ylabel("Objective Value")
                plt.title("Clipped vs Unclipped Surrogate Objective")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"logs/ppo_clip_vs_unclip_ep{episode}.png")
                plt.close()

        if episode % 10 == 0:
            agent.save_checkpoint()
        print(f"Episode {episode}, Rewards per env: {total_reward.tolist()}")
        # success_rate = (total_reward > threshold).float().mean()
        # print(f"Success Rate: {success_rate.item()*100:.2f}%")



        agent.logged_ratios.clear()
        agent.logged_advantages.clear()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load_path", type=str, nargs='?', default=None, help="Path for loading model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64, help="Hidden dimension for the network")
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_ppo(args)