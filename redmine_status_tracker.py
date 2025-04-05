#!/usr/bin/env python3
import requests
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import configparser
import os
import sys
import json
import logging
from collections import defaultdict
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('redmine_tracker.log')
    ]
)

class RedmineStatusTracker:
    def __init__(self, api_key, url="http://redmine.example.org"):
        """
        Initialize the Redmine status tracker
        
        Args:
            api_key (str): Your Redmine API key
            url (str): URL of your Redmine instance
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.headers = {'X-Redmine-API-Key': self.api_key, 'Content-Type': 'application/json'}
        
    def get_ticket_journals(self, ticket_id):
        """
        Get all journal entries (history) for a ticket
        
        Args:
            ticket_id (int): The Redmine ticket ID
            
        Returns:
            dict: Ticket data including journals
        """
        try:
            logging.info(f"Retrieving ticket {ticket_id} with journals...")
            
            url = f"{self.url}/issues/{ticket_id}.json"
            params = {'include': 'journals'}
            
            logging.info(f"Request URL: {url}")
            logging.info(f"Request params: {params}")
            
            response = requests.get(
                url,
                headers=self.headers,
                params=params
            )
            
            logging.info(f"Response status code: {response.status_code}")
            
            response.raise_for_status()
            
            # Log raw response in debug level (to avoid excessive logging for large tickets)
            json_response = response.json()
            logging.debug(f"Raw response: {json.dumps(json_response, indent=2)}")
            
            # Log summary of the response
            if 'issue' in json_response:
                issue = json_response['issue']
                journal_count = len(issue.get('journals', []))
                logging.info(f"Retrieved ticket #{ticket_id}: '{issue.get('subject', 'No subject')}'")
                logging.info(f"Current status: {issue.get('status', {}).get('name', 'Unknown')}")
                logging.info(f"Journal entries: {journal_count}")
                
                # Log the first few journal entries as a sample
                if journal_count > 0:
                    sample_size = min(3, journal_count)
                    logging.info(f"Sample of the first {sample_size} journal entries:")
                    for i in range(sample_size):
                        journal = issue['journals'][i]
                        logging.info(f"  #{i+1} - {journal.get('created_on')} by {journal.get('user', {}).get('name', 'Unknown')}")
                        for detail in journal.get('details', []):
                            logging.info(f"    - Changed '{detail.get('name')}' from '{detail.get('old_value', 'None')}' to '{detail.get('new_value', 'None')}'")
            
            return json_response
        except requests.exceptions.RequestException as e:
            logging.error(f"Error retrieving ticket {ticket_id}: {e}")
            return None
    
    def extract_status_changes(self, ticket_data):
        """
        Extract status changes from ticket journal entries
        
        Args:
            ticket_data (dict): Ticket data with journal entries
            
        Returns:
            list: List of status change events with timestamps
        """
        if not ticket_data or 'issue' not in ticket_data:
            return []
        
        status_changes = []
        
        # Process journals for status changes
        for journal in ticket_data['issue'].get('journals', []):
            for detail in journal.get('details', []):
                if detail.get('name') == 'status_id':
                    status_id = int(detail.get('new_value'))
                    # Get status name if possible
                    status_name = self.get_status_name(status_id)
                    
                    status_changes.append({
                        'timestamp': datetime.fromisoformat(journal['created_on'].replace('Z', '+00:00')),
                        'status_id': status_id,
                        'status_name': status_name
                    })
        
        return sorted(status_changes, key=lambda x: x['timestamp'])
    
    def extract_assignment_events(self, ticket_data):
        """
        Extract assignment events from ticket journal entries
        
        Args:
            ticket_data (dict): Ticket data with journal entries
            
        Returns:
            list: List of assignment events with timestamps and details
        """
        assignment_events = []
        for journal in ticket_data['issue'].get('journals', []):
            for detail in journal.get('details', []):
                if detail.get('name') == 'assigned_to_id':
                    assignment_events.append({
                        'timestamp': datetime.fromisoformat(journal['created_on'].replace('Z', '+00:00')),
                        'old_assigned': detail.get('old_value'),
                        'new_assigned': detail.get('new_value')
                    })
        return sorted(assignment_events, key=lambda x: x['timestamp'])
    
    def get_status_name(self, status_id):
        """
        Get status name from status ID (you might want to cache this)
        
        Args:
            status_id (int): The status ID
            
        Returns:
            str: The status name
        """
        try:
            logging.info(f"Looking up name for status ID: {status_id}")
            
            url = f"{self.url}/issue_statuses.json"
            logging.info(f"Request URL: {url}")
            
            response = requests.get(url, headers=self.headers)
            
            logging.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                json_response = response.json()
                logging.debug(f"Raw status response: {json.dumps(json_response, indent=2)}")
                
                # Modified section: Search through issue_statuses list
                statuses = json_response.get('issue_statuses', [])
                matched_status = next(
                    (s for s in statuses if s.get('id') == status_id),
                    None
                )
                
                if matched_status:
                    return matched_status.get('name', f"Status {status_id}")
                else:
                    return f"Status {status_id}"
            else:
                logging.warning(f"Could not retrieve status name for ID {status_id}, response code: {response.status_code}")
                return f"Status {status_id}"
        except Exception as e:
            logging.error(f"Error retrieving status name for ID {status_id}: {e}")
            return f"Status {status_id}"
    
    def calculate_time_in_statuses(self, status_changes):
        """
        Calculate time spent in each status
        
        Args:
            status_changes (list): List of status change events
            
        Returns:
            dict: Time spent in each status in seconds
        """
        if not status_changes:
            return {}
        
        time_in_status = defaultdict(float)
        
        # Add current time as the final timestamp if ticket is still open
        if len(status_changes) > 0:
            current_status = status_changes[-1]['status_name']
            now = datetime.now(tz=status_changes[-1]['timestamp'].tzinfo).replace(microsecond=0)
            status_changes.append({
                'timestamp': now,
                'status_id': status_changes[-1]['status_id'],
                'status_name': current_status
            })
        
        # Calculate time spent in each status
        for i in range(len(status_changes) - 1):
            current = status_changes[i]
            next_change = status_changes[i + 1]
            duration = (next_change['timestamp'] - current['timestamp']).total_seconds()
            time_in_status[current['status_name']] += duration
            logging.info(f"Duration {duration} for status {current['status_name']}, with initial timestamp {current['timestamp']} and final timestamp {next_change['timestamp']} , for a total of {time_in_status[current['status_name']]}")
            
        return dict(time_in_status)
    
    def format_time_duration(self, seconds):
        """
        Format time duration in a human-readable format
        
        Args:
            seconds (float): Time duration in seconds
            
        Returns:
            str: Formatted time string
        """
        days = seconds // (24 * 3600)
        remaining = seconds % (24 * 3600)
        hours = remaining // 3600
        remaining %= 3600
        minutes = remaining // 60
        
        if days > 0:
            return f"{int(days)}d {int(hours)}h {int(minutes)}m"
        elif hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        else:
            return f"{int(minutes)}m"
    
    def generate_time_report(self, ticket_ids):
        """
        Generate a report of time spent in each status for multiple tickets
        
        Args:
            ticket_ids (list): List of ticket IDs
            
        Returns:
            dict: Report data for each ticket
        """
        report = {}
        
        for ticket_id in ticket_ids:
            ticket_data = self.get_ticket_journals(ticket_id)
            if not ticket_data:
                print(f"Skipping ticket {ticket_id} - unable to retrieve data")
                continue
                
            status_changes = self.extract_status_changes(ticket_data)
            time_in_statuses = self.calculate_time_in_statuses(status_changes)
            
            # Convert seconds to readable format
            formatted_times = {status: self.format_time_duration(seconds) 
                              for status, seconds in time_in_statuses.items()}
            
            # Calculate additional time metrics
            created_on = datetime.fromisoformat(ticket_data['issue']['created_on'].replace('Z', '+00:00'))
            
            # FIX: Extract and store estimated hours directly from the API response instead of calculating
            # This fixes the issue with the Original Estimation calculation
            estimated_hours = ticket_data['issue'].get('estimated_hours')
            if estimated_hours is not None:
                estimated_seconds = estimated_hours * 3600  # Convert hours to seconds
            else:
                # Fallback to calculating from start_date and due_date if estimated_hours is not available
                try:
                    start_date = datetime.strptime(ticket_data['issue'].get('start_date', ''), "%Y-%m-%d")
                    due_date = datetime.strptime(ticket_data['issue'].get('due_date', ''), "%Y-%m-%d")
                    # Calculate business days between start and due date (excluding weekends)
                    business_days = sum(1 for d in range((due_date - start_date).days + 1) 
                                       if (start_date + timedelta(days=d)).weekday() < 5)
                    estimated_seconds = business_days * 8 * 3600  # 8 hours per business day
                except (ValueError, TypeError):
                    # If dates are missing or invalid, set estimation to None
                    estimated_seconds = None
    
            assignment_events = self.extract_assignment_events(ticket_data)
            if assignment_events:
                assignment_time = assignment_events[0]['timestamp']
                time_to_assign = (assignment_time - created_on).total_seconds()
            else:
                # If no assignment event, assume ticket was assigned at creation
                assignment_time = created_on
                time_to_assign = 0
            
            time_new_to_in_progress = None
            # Calculate time from assignment (or creation if assigned at creation) to first "In Progress" event
            for event in status_changes:
                if event['timestamp'] > assignment_time and event['status_name'] == "In Progress":
                    time_new_to_in_progress = (event['timestamp'] - assignment_time).total_seconds()
                    break
            
            report[ticket_id] = {
                'subject': ticket_data['issue'].get('subject', f"Ticket {ticket_id}"),
                'status_history': status_changes,
                'time_in_statuses': time_in_statuses,
                'formatted_times': formatted_times,
                'time_to_assign': time_to_assign,
                'formatted_time_to_assign': self.format_time_duration(time_to_assign) if time_to_assign is not None else "N/A",
                'time_new_to_in_progress': time_new_to_in_progress,
                'formatted_time_new_to_in_progress': self.format_time_duration(time_new_to_in_progress) if time_new_to_in_progress is not None else "N/A",
                'original_estimation': estimated_seconds,
                'formatted_time_original_estimation': self.format_time_duration(estimated_seconds) if estimated_seconds is not None else "N/A"
            }
            
        return report
    
    def plot_time_in_statuses(self, report, focus_status=None, output_dir='.'):
        """
        Generate visualization of time spent in statuses
        
        Args:
            report (dict): Report data
            focus_status (str, optional): Status to highlight in the charts
            output_dir (str): Directory to save the plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        all_tickets_data = []
        
        for ticket_id, data in report.items():
            for status, seconds in data['time_in_statuses'].items():
                hours = seconds / 3600
                all_tickets_data.append({
                    'Ticket ID': str(ticket_id),
                    'Subject': data['subject'],
                    'Status': status,
                    'Hours': hours,
                    'Estimation': data['original_estimation']
                })
        
        if not all_tickets_data:
            print("No data to plot.")
            return
            
        df = pd.DataFrame(all_tickets_data)
        
        # 1. Bar chart of time spent by status across all tickets
        plt.figure(figsize=(12, 6))
        status_summary = df.groupby('Status')['Hours'].sum().reset_index()
        status_summary = status_summary.sort_values('Hours', ascending=False)
        
        colors = ['#1f77b4'] * len(status_summary)
        if focus_status:
            colors = ['#ff7f0e' if status == focus_status else '#1f77b4' for status in status_summary['Status']]
            
        sns.barplot(x='Status', y='Hours', data=status_summary, palette=colors)
        plt.title('Total Time Spent in Each Status Across All Tickets')
        plt.xlabel('Status')
        plt.ylabel(['Hours'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/status_time_summary.png")
        
        # 2. Per-ticket status time breakdown
        if len(report) <= 30:  # Only create this plot if we have 10 or fewer tickets
            plt.figure(figsize=(14, 8))
            ticket_pivot = df.pivot_table(
                index='Ticket ID', 
                columns='Status', 
                values='Hours', 
                aggfunc='sum'
            ).fillna(0)
            
            ticket_pivot.plot(kind='bar', stacked=True, figsize=(14, 8))
            plt.title('Time Spent in Each Status by Ticket')
            plt.xlabel('Ticket ID')
            plt.ylabel('Hours')
            plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/ticket_status_breakdown.png")
    
        # 3. If we have a focus status, show comparison across tickets
        if focus_status and focus_status in df['Status'].values:
            plt.figure(figsize=(12, 6))
            focus_df = df[df['Status'] == focus_status]
            focus_df = focus_df.sort_values('Hours', ascending=False)
            
            # Reshape the DataFrame from wide to long format
            focus_melted = focus_df.melt(id_vars=['Ticket ID'], 
                                        value_vars=['Hours', 'Estimation'], 
                                        var_name='Metric', 
                                        value_name='Value')
            
            # Create a barplot with side-by-side bars for Hours and Estimation
            sns.barplot(x='Ticket ID', y='Value', hue='Metric', data=focus_melted, dodge=True)
            plt.title(f'Time Spent in "{focus_status}" Status by Ticket')
            plt.xlabel('Ticket IDs')
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/focus_status_comparison.png")

            
        # 4. Generate a timeline visualization for a single ticket if only one was provided
        if len(report) == 1:
            ticket_id = list(report.keys())[0]
            self._plot_ticket_timeline(ticket_id, report[ticket_id], output_dir)
            
        print(f"Charts saved to {output_dir}")
            
    def _plot_ticket_timeline(self, ticket_id, ticket_data, output_dir):
        """
        Create a timeline visualization for a single ticket
        
        Args:
            ticket_id (int): Ticket ID
            ticket_data (dict): Ticket report data
            output_dir (str): Directory to save the plot
        """
        status_history = ticket_data['status_history']
        if len(status_history) < 2:
            return
            
        # Prepare timeline data
        timeline_data = []
        for i in range(len(status_history) - 1):
            current = status_history[i]
            next_change = status_history[i + 1]
            
            timeline_data.append({
                'Status': current['status_name'],
                'Start': current['timestamp'],
                'End': next_change['timestamp'],
                'Duration': (next_change['timestamp'] - current['timestamp']).total_seconds() / 3600  # hours
            })
        
        # Create the timeline plot
        plt.figure(figsize=(15, 6))
        
        # Generate a color map for statuses
        statuses = sorted(set(item['Status'] for item in timeline_data))
        colors = plt.cm.tab10(np.linspace(0, 1, len(statuses)))
        status_colors = {status: colors[i] for i, status in enumerate(statuses)}
        
        # Plot the timeline
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        for i, event in enumerate(timeline_data):
            plt.barh(
                y_pos, 
                event['Duration'], 
                left=mdates.date2num(event['Start']),
                color=status_colors[event['Status']],
                edgecolor='black'
            )
            
            # Add status label if it's the first occurrence or different from previous
            if i == 0 or timeline_data[i-1]['Status'] != event['Status']:
                y_ticks.append(y_pos)
                y_labels.append(event['Status'])
            
            y_pos += 1
        
        # Format the plot
        plt.yticks(y_ticks, y_labels)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.title(f"Timeline for Ticket #{ticket_id}: {ticket_data['subject']}")
        plt.xlabel('Date')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ticket_{ticket_id}_timeline.png")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Track time spent in Redmine ticket statuses')
    parser.add_argument('ticket_ids', nargs='+', type=int, help='Redmine ticket ID(s)')
    parser.add_argument('--config', type=str, default='config.ini', help='Path to config file')
    parser.add_argument('--focus', type=str, help='Status to focus on in reports (e.g. "In Progress")')
    parser.add_argument('--output', type=str, default='reports', help='Output directory for reports and charts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dump-raw', action='store_true', help='Dump raw JSON responses to files')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Verbose logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Load configuration
    config = configparser.ConfigParser()
    
    if not os.path.exists(args.config):
        print(f"Creating config file at {args.config}")
        config['DEFAULT'] = {
            'api_key': 'YOUR_API_KEY_HERE',
            'url': 'http://redmine.example.org'
        }
        with open(args.config, 'w') as f:
            config.write(f)
        print(f"Please edit {args.config} to add your Redmine API key and URL.")
        sys.exit(1)
        
    config.read(args.config)
    api_key = config['DEFAULT']['api_key']
    url = config['DEFAULT']['url']
    
    if api_key == 'YOUR_API_KEY_HERE':
        print(f"Please edit {args.config} to add your Redmine API key.")
        sys.exit(1)
    
    # Create tracker and generate report
    logging.info(f"Initializing Redmine Status Tracker with URL: {url}")
    tracker = RedmineStatusTracker(api_key, url)
    
    logging.info(f"Generating report for ticket IDs: {args.ticket_ids}")
    report = tracker.generate_time_report(args.ticket_ids)
    
    # If dump-raw flag is set, save the raw report data to a JSON file
    if args.dump_raw:
        dump_dir = os.path.join(args.output, 'raw_data')
        os.makedirs(dump_dir, exist_ok=True)
        
        dump_file = os.path.join(dump_dir, f"raw_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logging.info(f"Dumping raw report data to: {dump_file}")
        
        # Create a serializable version of the report
        serializable_report = {}
        for ticket_id, data in report.items():
            serializable_report[str(ticket_id)] = {
                'subject': data['subject'],
                'time_in_statuses': data['time_in_statuses'],
                'formatted_times': data['formatted_times'],
                'time_to_assign': data['time_to_assign'],
                'formatted_time_to_assign': data['formatted_time_to_assign'],
                'time_new_to_in_progress': data['time_new_to_in_progress'],
                'formatted_time_new_to_in_progress': data['formatted_time_new_to_in_progress'],
                'original_estimation': data['original_estimation'],
                'formatted_time_original_estimation': data['formatted_time_original_estimation'],
                'status_history': [
                    {
                        'timestamp': change['timestamp'].isoformat(),
                        'status_id': change['status_id'],
                        'status_name': change['status_name']
                    }
                    for change in data['status_history']
                ]
            }
        
        with open(dump_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
    
    # Print text report
    print("\n=== Redmine Status Time Tracking Report (Business Hours Only) ===\n")
    print("Working hours: 8 hours per day (Mon-Fri excluding holidays)\n")
    
    for ticket_id, data in report.items():
        print(f"Ticket #{ticket_id}: {data['subject']}")
        print("-" * 40)
        for status, time_str in data['formatted_times'].items():
            highlight = "*" if args.focus and status == args.focus else " "
            print(f"{highlight} {status}: {time_str}")
        print(f"Time to assign: {data['formatted_time_to_assign']}")
        print(f"Time from assignment to 'In Progress': {data['formatted_time_new_to_in_progress']}")
        print(f"Original Estimation: {data['formatted_time_original_estimation']}")
        print()
        
    # Generate charts
    logging.info(f"Generating visualizations with focus on status: {args.focus}")
    tracker.plot_time_in_statuses(report, args.focus, args.output)
    
    logging.info("Script execution completed successfully")
    
if __name__ == "__main__":
    try:
        logging.info("Starting Redmine Status Tracker")
        main()
    except Exception as e:
        logging.exception(f"Unhandled exception: {e}")
        print(f"An error occurred: {e}. See redmine_tracker.log for details.")
        sys.exit(1)